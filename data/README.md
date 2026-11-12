# data/

Not committed (about 200 MB). Rebuild it with:

```bash
bash scripts/get_data.sh     # data/raw/AgentDrift  (pinned commit 014a514)
bash scripts/run_week1.sh    # data/processed/      (anonymized traces, decision points, splits)
```

Source: [AgentDrift](https://github.com/Asif-0209/AgentDrift), CC BY 4.0. 12,536 synthetic agent traces
(71,024 steps) over five domains: banking, coding, email, medical, web. Each step is JSON with the agent's
thought, tool name, tool arguments, tool result and a label (benign, injection_point, hijacked, failed_injection).
