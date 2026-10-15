# RescueGroups exploration

This is an isolated feasibility/audit branch for RescueGroups listings. It is separate
from the Austin shelter analysis and does not feed the final report.

- `download_available_listings.py` downloads resumable raw listing pages and writes a manifest.
- `audit_adopted_history.py` samples adopted records and writes feasibility reports about text richness and temporal-leakage risk.
- Related downloaded snapshots and reports live under `data/rescuegroups_adopted_feasibility/`.

These scripts may require API credentials and network access. Run them only when the
RescueGroups audit is being updated.
