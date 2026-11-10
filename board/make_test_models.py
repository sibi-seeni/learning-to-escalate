"""Make tiny stand-in models for the week 1 board check. Run on your laptop.

  board/models/tiny_gru.onnx        1-layer GRU (hidden 64) over an 11-step prefix -> P(hijacked)
  board/models/tiny_gru_int8.onnx   same, run through int8 dynamic quantization (GRU itself stays float, see below)
  board/models/tiny_lstm.onnx       1-layer LSTM, same shape
  board/models/tiny_lstm_int8.onnx  same, LSTM weights really int8
  board/models/tiny_lgbm.txt        LightGBM, 100 trees on 20 features (deferrer stand-in)
  board/models/expected.npz         fixed inputs + laptop outputs, so the board can check it gets the same numbers

The weights are random. This only proves the runtimes install and run on the UNO Q, and gives a first
latency number. Real models replace these in week 2.

Run:  python board/make_test_models.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "models"
SEQ, FEAT, HID = 11, 64, 64  # max trace length in AgentDrift is 11 steps
RNN_MODELS = ("tiny_gru", "tiny_gru_int8", "tiny_lstm", "tiny_lstm_int8")


def build_rnn(path, cell="GRU"):
    """cell='GRU' (3 gates) or 'LSTM' (4 gates). Same input/output either way."""
    gates = 3 if cell == "GRU" else 4
    rng = np.random.RandomState(0)
    init = [
        numpy_helper.from_array((rng.randn(1, gates * HID, FEAT) * 0.1).astype(np.float32), "W"),
        numpy_helper.from_array((rng.randn(1, gates * HID, HID) * 0.1).astype(np.float32), "R"),
        numpy_helper.from_array(np.zeros((1, 2 * gates * HID), np.float32), "B"),
        numpy_helper.from_array((rng.randn(HID, 1) * 0.1).astype(np.float32), "W_out"),
        numpy_helper.from_array(np.zeros((1,), np.float32), "b_out"),
        numpy_helper.from_array(np.array([1, 0, 2], np.int64), "perm"),
        numpy_helper.from_array(np.array([-1, HID], np.int64), "shape"),
    ]
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], perm=[1, 0, 2]),            # [batch, seq, feat] -> [seq, batch, feat]
        helper.make_node(cell, ["x_t", "W", "R", "B"], ["", "h"], hidden_size=HID),  # h: [1, batch, hid]
        helper.make_node("Reshape", ["h", "shape"], ["h2"]),                      # [batch, hid]
        helper.make_node("MatMul", ["h2", "W_out"], ["logit0"]),
        helper.make_node("Add", ["logit0", "b_out"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["p"]),
    ]
    graph = helper.make_graph(
        nodes, f"tiny_{cell.lower()}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", SEQ, FEAT])],
        [helper.make_tensor_value_info("p", TensorProto.FLOAT, ["batch", 1])],
        initializer=[i for i in init if i.name != "perm"],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # readable by older onnxruntime builds too
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    OUT.mkdir(exist_ok=True)
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for cell in ("GRU", "LSTM"):
        name = f"tiny_{cell.lower()}"
        build_rnn(OUT / f"{name}.onnx", cell)
        quantize_dynamic(str(OUT / f"{name}.onnx"), str(OUT / f"{name}_int8.onnx"), weight_type=QuantType.QInt8)

    rng = np.random.RandomState(1)
    X, y = rng.randn(2000, 20), rng.randint(0, 2, 2000)
    booster = lgb.train({"objective": "binary", "num_leaves": 15, "verbose": -1}, lgb.Dataset(X, y), num_boost_round=100)
    booster.save_model(str(OUT / "tiny_lgbm.txt"))

    x_seq = rng.randn(8, SEQ, FEAT).astype(np.float32)
    x_tab = rng.randn(8, 20)
    expected = {"x_seq": x_seq, "x_tab": x_tab, "lgbm": booster.predict(x_tab)}
    for name in RNN_MODELS:
        sess = ort.InferenceSession(str(OUT / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        expected[name] = sess.run(None, {"x": x_seq})[0]
    np.savez(OUT / "expected.npz", **expected)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:22s} {f.stat().st_size / 1024:8.1f} KB")
    for cell in ("gru", "lstm"):
        print(f"{cell}: int8 vs float max |diff| = {float(np.abs(expected[f'tiny_{cell}'] - expected[f'tiny_{cell}_int8']).max()):.4f}")
    print("Note: onnxruntime's dynamic int8 quantizer supports LSTM but NOT GRU, so tiny_gru_int8 is still a float GRU "
          "(only the tiny output layer is int8). See docs/WEEK1.md.")


if __name__ == "__main__":
    main()
