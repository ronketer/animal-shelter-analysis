# Retained RescueGroups audit utilities

This optional listing-text direction was investigated and excluded from the final
models because pre-adoption versions of descriptions could not be verified.
The [saved feasibility check](../optional/rescuegroups_feasibility_report.md) supports
the discussion in the final report.

`tests/test_rescuegroups_audit.py` imports `audit_adopted_history.py`, which in turn
imports shared request and credential helpers from `download_available_listings.py`.
Both modules are retained to preserve these offline tests. Tests use synthetic records
and make no API requests. The utilities are not part of Austin/Sonoma reproduction.

Executing either script directly can contact RescueGroups and requires a local API key.
Downloaded records belong under the ignored `data/` directory; credentials must remain
local. The larger legacy audit is distinct from the smaller saved feasibility check in
`optional/`.
