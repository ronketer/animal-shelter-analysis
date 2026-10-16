# RescueGroups listing-text feasibility

This independent experiment asked whether adopted listings could support a historical
study of listing text and adoption delays. It does not feed the final Austin/Sonoma
analyses or Streamlit demo.

The [saved results](rescuegroups_feasibility_report.md) support the final report's
decision to exclude this direction: only 92 of 952 sampled records (9.7%) had usable
text and a valid available-to-adopted duration. Current descriptions could not be
verified as the versions available before adoption. Record-level update timestamps
indicate risk, not proof that description text changed.

To repeat the check, run from the repository root after the normal project setup:

```sh
uv run python experiments/rescuegroups/rescuegroups_feasibility_check.py
```

The script prompts securely for a RescueGroups API key, or reads
`RESCUE_GROUPS_API_KEY` from the environment. It makes live API requests; no credentials
or network access are needed for the core project or its tests. New aggregate Markdown
and JSON outputs go to ignored `data/rescuegroups_feasibility/`, preserving the saved
report here. The API population changes, so a new run will not reproduce the historical
sample exactly.
