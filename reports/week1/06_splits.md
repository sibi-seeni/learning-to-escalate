# Week 1 · Shift splits

## Tool coverage (hold out 20% of each domain's tools)

| seed | held-out tools | train DPs kept | train hijacked | smallest domain (train DPs) | test N per level | test hijacked rate |
|---|---|---|---|---|---|---|
| 0 | 21 | 17062 / 51507 | 1800 | email (771) | 2818 | 0.155 |
| 1 | 21 | 26044 / 51507 | 3244 | banking (1733) | 2560 | 0.143 |
| 2 | 21 | 23625 / 51507 | 2921 | web (3575) | 2923 | 0.123 |

Held-out tools, seed 0: apply_loan, close_tab, compare_prices, create_referral, delete_email, fill_form, forward_email, get_allergies, get_balance, get_medical_history, get_patient_record, get_rewards, get_statement, git_commit, git_log, git_pull, list_files, open_issue, search_inbox, submit_form, summarize_thread

## Leave one domain out

| held-out domain | train DPs | val DPs | test DPs | test hijacked |
|---|---|---|---|---|
| banking | 43081 | 8238 | 11634 | 1170 |
| coding | 40955 | 7682 | 14817 | 2808 |
| email | 40663 | 7591 | 14986 | 2862 |
| medical | 40882 | 7668 | 14795 | 2758 |
| web | 40447 | 7705 | 14792 | 2751 |
