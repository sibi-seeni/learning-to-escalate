# Week 1 · Data summary

AgentDrift commit `014a514fa998b4ac4519579fceb8a5884b379bda`

### stratified split

| part | trajectories | steps | attacked | benign | failed_attack | hard_negative |
|---|---|---|---|---|---|---|
| train | 8775 | 49750 | 3875 | 2800 | 1050 | 1050 |
| val | 1880 | 10722 | 830 | 600 | 225 | 225 |
| test | 1881 | 10552 | 831 | 600 | 225 | 225 |

Test part by domain: banking 321, coding 390, email 390, medical 390, web 390
Test attacked by compliance: delayed_execution 134, full_hijack 466, partial_hijack 231

### taskdisjoint split

| part | trajectories | steps | attacked | benign | failed_attack | hard_negative |
|---|---|---|---|---|---|---|
| train | 9081 | 51507 | 4008 | 2891 | 1089 | 1093 |
| val | 1733 | 9721 | 753 | 584 | 193 | 203 |
| test | 1722 | 9796 | 775 | 525 | 218 | 204 |

Test part by domain: banking 317, coding 382, email 343, medical 376, web 304
Test attacked by compliance: delayed_execution 117, full_hijack 450, partial_hijack 208

