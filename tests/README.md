# Tests

Run from the repository root:

```sh
uv run python -m unittest discover -s tests -v
```

The suite uses synthetic fixtures for joins, intake features, shelter-load event-ledger
edge cases, animal-grouped splits, model utilities, spatial
statistics, and geocode merge keys. It also checks inference with the supplied frozen
demo artifact, including its final feature schema, scikit-learn compatibility, and
four-class probability contract.

No raw Austin/Sonoma data, API credentials, network requests, or model training are
required. The standalone experiment under `experiments/rescuegroups/` is separate
from this suite. Full analysis reproduction requires the original snapshots described in the root README.
