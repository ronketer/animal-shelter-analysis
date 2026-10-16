# Streamlit loading diagnostics

Measured on 2026-10-16. The intermittent minutes-long loading screen was **not
reproduced**. A real session on the deployed app completed successfully. There is
no evidence here that dependency installation, model loading, or charts cause the
reported delay, so Python, dependencies, theme, inference, and caching are unchanged.

## Measurements

Local environment: Windows, Python 3.13.5, scikit-learn 1.9.0, Streamlit 1.62.0,
joblib 1.5.3, NumPy 2.5.1, pandas 3.0.3, Altair 6.2.2, SciPy 1.18.0. Python and
scikit-learn match the artifact metadata. The locked research environment remains
intact. Research/data-loading modules are not imported by the dashboard.

| Stage | Baseline | With diagnostic logging |
| --- | ---: | ---: |
| Sequential Altair/pandas/Streamlit/demo_service imports, fresh interpreter | 3.730 s | 2.355 s |
| Metadata JSON read/parse | 0.509 ms | 0.188 ms |
| Classifier `joblib.load` after imports | 0.580 s | 0.183 s |
| First inference, standalone fresh interpreter | 2.400 s | 1.408 s |
| Subsequent inference across four representative profiles | 31.6–36.8 ms | 32.7–35.3 ms |
| Initial dashboard AppTest execution, framework/service already imported | 2.405 s | 2.342 s |
| Unchanged-input AppTest rerun | 0.116 s | 0.123 s |
| Initial Altair serialization/enqueue within AppTest | 36.5 ms | 24.5 ms |
| Real local WebSocket session: first script completion | 7.358 s | 5.812 s |
| Same local WebSocket: next script completion | 0.215 s | 0.211 s |

These are individual observations, not benchmark medians. Filesystem cache,
library initialization, and system load vary; **no speedup is attributed to logging**.
In the final fresh local server, process launch to healthy HTTP response took
1.077 s, before the first app session. Its app-side first-run timings were:
imports 1.932 s, bundle 0.369 s (model/metadata loader 0.367 s), inference 1.437 s,
chart construction plus serialization/enqueue 0.051 s, total app execution 4.477 s.
WebSocket completion includes additional session/framework time.

The deployed embedded application at `/~/+/` returned its HTML shell in 0.407 s,
plain-text health in 0.300 s, and JSON host configuration in 0.200 s. Its
`/~/+/_stcore/stream` WebSocket connected in 0.338 s, delivered a first message in
0.220 s, and completed the first requested run in 5.832 s and the next in 0.380 s.
Both runs sent all four metrics, input widgets, and a Vega-Lite chart with successful
`script_finished` status and no exception element. This was the first session
observed by this probe, not a controlled Cloud cold boot.

The top-level URL serves Community Cloud's outer web page (`/-/build/assets/...`).
Requests to top-level `/_stcore/health` or `/_stcore/stream` do not test the embedded
application; the initial top-level WebSocket probe failed for this reason. Inspect
the actual iframe route in DevTools rather than treating this as an app failure.

## What was verified

- The existing `st.cache_resource` loads the bundle once across reruns. No TTL or
  cache invalidation loop was found. Comparison inference runs only when enabled.
- Streamlit AppTest verified initial UI generation, an unchanged-input rerun,
  changing species, enabling comparison, and changing comparison age. All displayed
  metrics and comparison tables matched the baseline exactly.
- Four representative profiles (dog, cat, injured dog, other species) produced 16
  identical before/after probabilities: maximum absolute difference **0.0**, checked
  with absolute tolerance `1e-12` and zero relative tolerance. The ten-feature
  contract, four class labels, and evaluation metadata were unchanged.
- All 18 existing tests passed. The saved artifact loaded with
  `InconsistentVersionWarning` treated as an error. A joblib/NumPy array-shape
  deprecation warning was non-fatal; no package change is justified by it.
- SHA-256 of `demo_artifacts/outcome_model.joblib`:
  `970DA7F0FF4E9B6004B7DC09F9DCE026CF8780735FA4528BA762F4FF408FA281`.
- SHA-256 of `demo_artifacts/demo_metadata.json`:
  `9835DE91E3A13344A69E9DED774F94FFCDA131BFA194BFF85476095B5219F4D9`.

Temporary profiling scripts, timings, and baseline probabilities are retained under
ignored `tmp/loading-diagnostics/`. They do not run during deployment. The classifier
was never fitted or exported.

## Capture the next slow Cloud load

No connected browser surface was available for automated browser inspection. HTTP
and real WebSocket/protobuf sessions were tested, but browser console errors,
JavaScript execution, actual chart painting, Cloud memory/CPU, and authenticated
deployment logs remain unobserved. Cloud wake-up/provisioning, wrapper readiness,
and browser connectivity are candidates, not confirmed root causes. The supplied
12-second clone-to-server log does not measure session establishment or app execution.

1. After Cloud deploys the latest `main`, open **Manage app → Logs**. Keep Python
   3.13, `src/demo_app.py`, and the existing dependency files. No environment rebuild
   or setting change is required for this diagnostic-only update.
2. In the browser, open DevTools **before** navigating, enable **Preserve log**, and
   record the navigation time. Watch Console errors and Network document, JS/CSS,
   Fetch/XHR, and WS requests, including the outer Cloud page and its app iframe.
   Check whether Cloud is waking/provisioning or whether the app iframe exists yet.
3. Find the actual app WebSocket (observed here at `/~/+/_stcore/stream`). It should
   upgrade with HTTP **101** and receive binary messages. Record handshake duration,
   time to first frame, failed handshakes, disconnects/reconnects, and stalled assets.
   HTTP 200 HTML from a wrapper URL is not an app health or WebSocket success.
4. Correlate server timestamps with `shelter_demo` log lines:
   - `initial_session imports_complete`: app imports have finished; duration includes
     already-cached module imports on later sessions.
   - `model_load started` / `model_load complete`: uncached metadata/model loading.
     Absence of these on a warm session is expected because of `st.cache_resource`.
   - `initial_session complete`: total Python script time, bundle retrieval,
     inference, and initial chart construction/serialization. This measures work
     sent to Streamlit, not browser paint. It logs once per successfully initialized
     session and contains no profile values or user identifiers.
5. Interpret the evidence: a long server-to-first-marker gap can be connection/
   scheduling delay or stalled imports; a long model-load interval identifies
   deserialization; large inference/chart timings identify app computation. If
   Python completes quickly but the browser still loads, inspect frame delivery,
   JavaScript errors, iframe loading, and asset requests. Compare a private window
   and another browser/network to isolate extensions, cookies, VPN/proxy, or filtering.
6. Save relevant timestamps and redacted errors before rebooting. If Cloud's wrapper
   stays stuck despite a healthy backend, reboot through Manage app and remeasure.
   A reboot clearing the symptom does not establish its root cause. Escalate a
   repeatable wrapper/transport failure to Streamlit with timestamps, app URL, and
   redacted network evidence. Do not share cookies, auth tokens, or raw HAR files.

Do not disable CORS/XSRF or change dependency/Python versions speculatively. The
remote dev-container configuration is for Codespaces (its Python 3.11 image is a
separate development setting), not the supplied Community Cloud Python 3.13 runtime;
it was preserved during Git synchronization.

Relevant official guidance: [remote loading troubleshooting](https://docs.streamlit.io/knowledge-base/deploy/remote-start),
[resource caching](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource),
and [rebooting Community Cloud apps](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/reboot-your-app).
