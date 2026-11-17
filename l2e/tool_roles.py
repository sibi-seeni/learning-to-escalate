"""What each tool does, described in domain-neutral roles.

The monitor's tool ids are names it memorized in training, so a tool from a new domain (web's `fill_form`)
means nothing to it. Roles describe the tool by what it can do instead, so `fill_form` looks like
`send_email`: both send data outside. In a real deployment these would come from the tool's manifest or
permissions (most agent frameworks make developers declare what a tool may touch); here they are written
by hand, once, for AgentDrift's 103 tools.

Roles (a tool can have several):
  reads_private   reads the user's own stored data (balance, records, inbox, files, contacts)
  reads_untrusted brings in content someone else wrote (web pages, incoming mail, docs, logs, search results);
                  this is where injected instructions arrive
  writes          changes the user's own state (records, files, folders, drafts, settings)
  sends_out       moves data to another person or system (email, forms, messages, pushes, claims)
  moves_money     moves or commits money
  runs_code       executes or installs code
  deletes         removes or cancels something
  changes_trust   changes who or what is trusted (contacts, payees, logins, filters, card locks)

Tools not in the table (a brand-new tool at deployment time) get roles from keywords in their name;
see step_features.roles_from_name().
"""

ROLE_NAMES = ["reads_private", "reads_untrusted", "writes", "sends_out", "moves_money", "runs_code", "deletes",
              "changes_trust"]

_R = {
    # banking
    "get_balance": "reads_private", "get_transactions": "reads_private", "get_statement": "reads_private",
    "get_beneficiaries": "reads_private", "get_rewards": "reads_private", "check_credit_score": "reads_private",
    "check_limit": "reads_private", "get_fees": "reads_untrusted", "get_interest_rate": "reads_untrusted",
    "convert_currency": "moves_money", "transfer_money": "moves_money sends_out", "pay_bill": "moves_money sends_out",
    "apply_loan": "moves_money sends_out writes", "open_savings": "moves_money writes",
    "add_beneficiary": "changes_trust writes", "delete_beneficiary": "deletes changes_trust",
    "dispute_transaction": "sends_out writes", "freeze_card": "changes_trust writes",
    "unfreeze_card": "changes_trust writes", "set_alert": "writes",
    # coding
    "read_file": "reads_private", "list_files": "reads_private", "git_diff": "reads_private", "git_log": "reads_private",
    "search_code": "reads_private", "read_logs": "reads_untrusted reads_private", "search_docs": "reads_untrusted",
    "check_ci": "reads_untrusted", "check_syntax": "reads_private", "debug_error": "reads_private runs_code",
    "write_file": "writes", "create_file": "writes", "format_code": "writes", "delete_file": "deletes writes",
    "execute_command": "runs_code", "run_tests": "runs_code", "install_package": "runs_code reads_untrusted",
    "git_pull": "reads_untrusted writes", "git_commit": "writes", "git_push": "sends_out",
    "create_pr": "sends_out", "open_issue": "sends_out", "close_issue": "writes",
    # email
    "read_email": "reads_untrusted reads_private", "get_thread": "reads_untrusted reads_private",
    "search_inbox": "reads_untrusted reads_private", "summarize_thread": "reads_untrusted reads_private",
    "get_attachment": "reads_untrusted", "download_attachment": "reads_untrusted writes",
    "get_contacts": "reads_private", "send_email": "sends_out", "reply_email": "sends_out",
    "forward_email": "sends_out", "schedule_email": "sends_out", "create_draft": "writes",
    "create_folder": "writes", "move_email": "writes", "mark_read": "writes", "mark_unread": "writes",
    "delete_email": "deletes", "create_filter": "changes_trust writes", "add_contact": "changes_trust writes",
    "unsubscribe": "sends_out",
    # medical
    "get_patient_record": "reads_private", "get_medical_history": "reads_private", "get_allergies": "reads_private",
    "get_lab_results": "reads_private", "get_prescriptions": "reads_private", "get_vitals": "reads_private",
    "get_appointments": "reads_private", "get_referrals": "reads_private", "get_insurance_info": "reads_private",
    "search_patients": "reads_private", "update_patient_record": "writes", "add_diagnosis": "writes",
    "record_vitals": "writes", "create_prescription": "writes sends_out", "order_lab_test": "sends_out",
    "create_referral": "sends_out", "schedule_appointment": "writes", "cancel_appointment": "deletes",
    "send_patient_message": "sends_out", "submit_claim": "sends_out moves_money",
    # web
    "search_web": "reads_untrusted", "read_webpage": "reads_untrusted", "extract_links": "reads_untrusted",
    "extract_table": "reads_untrusted", "translate_page": "reads_untrusted", "compare_prices": "reads_untrusted",
    "check_availability": "reads_untrusted", "take_screenshot": "reads_untrusted",
    "copy_text": "reads_untrusted reads_private", "download_file": "reads_untrusted writes",
    "click_link": "reads_untrusted", "open_tab": "reads_untrusted", "close_tab": "writes", "go_back": "reads_untrusted",
    "scroll_page": "reads_untrusted", "bookmark_page": "writes", "fill_form": "sends_out",
    "submit_form": "sends_out", "login_site": "changes_trust sends_out", "logout_site": "changes_trust",
}
TOOL_ROLES = {tool: sorted(roles.split()) for tool, roles in _R.items()}
assert all(r in ROLE_NAMES for rs in TOOL_ROLES.values() for r in rs)
