"""Focused Streamlit dashboard for the frozen Austin shelter outcome model.

Run from repository root: uv run streamlit run src/demo_app.py
"""
from __future__ import annotations

import calendar
from time import perf_counter

_run_started = perf_counter()

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.logger import get_logger

from demo_service import load_demo_model, predict_intake

_imports_seconds = perf_counter() - _run_started
_logger = get_logger(__name__)

st.set_page_config(page_title="Shelter Outcome Explorer", page_icon="🐾", layout="wide",
                   initial_sidebar_state="collapsed")

_initial_session = not st.session_state.get("_initial_timing_logged", False)
if _initial_session:
    _logger.info("shelter_demo initial_session imports_complete seconds=%.3f", _imports_seconds)

OUTCOMES = ["Adoption", "Return to Owner", "Transfer", "Other"]
PRESETS = {
    "Stray dog": dict(animal_type="Dog", sex="Male", fixed="Intact", age_years=2.0,
                      intake_type="Stray", condition_group="Normal", is_black=False, month=6),
    "Surrendered cat": dict(animal_type="Cat", sex="Female", fixed="Altered", age_years=4.0,
                           intake_type="Owner Surrender", condition_group="Normal", is_black=True, month=6),
    "Injured stray dog": dict(animal_type="Dog", sex="Female", fixed="Unknown", age_years=6.0,
                             intake_type="Stray", condition_group="Injured", is_black=False, month=6),
}

st.markdown("""<style>
.block-container {max-width: 1300px; padding-top: 4rem; padding-bottom: 2rem;}
h1 {letter-spacing: -.035em;}
[data-testid="stMetric"] {background: rgba(120,140,160,.08); border: 1px solid rgba(120,140,160,.17);
 border-radius: 12px; padding: 9px 14px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
.small-note {opacity:.76; font-size:.89rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the saved model…")
def get_bundle():
    # This runs only on a resource-cache miss, shared across sessions and reruns.
    started = perf_counter()
    _logger.info("shelter_demo model_load started")
    bundle = load_demo_model()
    _logger.info("shelter_demo model_load complete seconds=%.3f", perf_counter() - started)
    return bundle


def reset_profile(preset: str) -> None:
    for key, value in PRESETS[preset].items():
        st.session_state[f"profile_{key}"] = value


def make_row(prefix: str, occupancy: float, intakes_7d: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "animal_type": st.session_state[f"{prefix}_animal_type"],
        "sex": st.session_state[f"{prefix}_sex"],
        "fixed": st.session_state[f"{prefix}_fixed"],
        "age_years": st.session_state[f"{prefix}_age_years"],
        "intake_type": st.session_state[f"{prefix}_intake_type"],
        "condition_group": st.session_state[f"{prefix}_condition_group"],
        "is_black": st.session_state[f"{prefix}_is_black"],
        "month": str(st.session_state[f"{prefix}_month"]),
        "occupancy": occupancy,
        "intakes_7d": intakes_7d,
    }])


def profile_editor(prefix: str) -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        st.selectbox("Species", ["Dog", "Cat", "Other"], key=f"{prefix}_animal_type")
        st.selectbox("Sex", ["Male", "Female", "Unknown"], key=f"{prefix}_sex")
        st.selectbox("Spay/neuter status", ["Intact", "Altered", "Unknown"], key=f"{prefix}_fixed")
        st.slider("Age (years)", 0.0, 20.0, step=0.5, key=f"{prefix}_age_years")
    with right:
        st.selectbox("Intake type",
                     ["Stray", "Owner Surrender", "Public Assist", "Abandoned", "Other"],
                     key=f"{prefix}_intake_type")
        st.selectbox("Intake condition",
                     ["Normal", "Injured", "Medical", "Behavior/Other"],
                     key=f"{prefix}_condition_group")
        st.selectbox("Intake month", list(range(1, 13)),
                     format_func=lambda n: calendar.month_name[n], key=f"{prefix}_month")
        st.checkbox("Black coat", key=f"{prefix}_is_black")


def chart(probs: pd.Series, title: str = "") -> None:
    data = pd.DataFrame({"Outcome": probs.index, "Probability": probs.values})
    # Keep zero in view while avoiding empty space for lower-probability profiles.
    max_probability = float(data["Probability"].max())
    axis_max = min(1.0, max(0.5, (int((max_probability + 0.14) * 10) + 1) / 10))
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=5, color="#38BDF8").encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, axis_max]),
                axis=alt.Axis(format="%")),
        y=alt.Y("Outcome:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("Outcome:N"), alt.Tooltip("Probability:Q", format=".1%")],
    )
    labels = alt.Chart(data).mark_text(align="left", dx=6, color="#E8F0F7").encode(
        x="Probability:Q", y=alt.Y("Outcome:N", sort="-x"),
        text=alt.Text("Probability:Q", format=".1%"))
    st.altair_chart((bars + labels).properties(height=215, title=title), width="stretch")


def compare_chart(a: pd.Series, b: pd.Series) -> None:
    data = pd.DataFrame({
        "Outcome": list(a.index) * 2,
        "Probability": list(a.values) + [float(b[k]) for k in a.index],
        "Profile": ["Original"] * len(a) + ["Comparison"] * len(a),
    })
    chart_obj = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Outcome:N", sort=list(a.index), title=None),
        yOffset="Profile:N",
        color=alt.Color("Profile:N", scale=alt.Scale(
            domain=["Original", "Comparison"], range=["#38BDF8", "#2DD4BF"])),
        tooltip=["Profile:N", "Outcome:N", alt.Tooltip("Probability:Q", format=".1%")],
    ).properties(height=250)
    st.altair_chart(chart_obj, width="stretch")


try:
    _bundle_started = perf_counter()
    bundle = get_bundle()
    _bundle_seconds = perf_counter() - _bundle_started
except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
    st.error("The frozen demo model or its metadata could not be loaded.")
    st.code("Get-ChildItem .\\demo_artifacts\\")
    st.caption(f"Details: {exc}")
    st.stop()

for field, value in PRESETS["Stray dog"].items():
    st.session_state.setdefault(f"profile_{field}", value)

st.caption("ANIMAL SHELTER ANALYSIS  /  INTERACTIVE RESEARCH DEMO")
st.title("🐾 Shelter Outcome Explorer")
st.write("Explore how intake-time characteristics relate to an Austin-trained "
         "classifier's predicted outcomes.")
st.caption("ℹ️ Research demonstration only — not a shelter decision or animal-care recommendation tool.")

m1, m2, m3 = st.columns(3)
m1.metric("Held-out macro-F1", f"{bundle.test_macro_f1:.3f}")
m2.metric("Outcome classes", "4")
m3.metric("Evaluation", "Animal-level split")

st.subheader("Explore an intake profile")
preset_col, reset_col = st.columns([3, 1], vertical_alignment="bottom")
with preset_col:
    preset = st.selectbox("Start from an example", list(PRESETS), key="preset_selection")
with reset_col:
    st.button("Load example", width="stretch",
              on_click=reset_profile, args=(preset,))

input_col, output_col = st.columns([1.15, 1], gap="large", vertical_alignment="top")
with input_col:
    with st.container(border=True):
        profile_editor("profile")
    with st.expander("Advanced · Shelter context", expanded=False):
        st.caption("These are inputs representing shelter conditions at intake. "
                   "Defaults come from the saved model metadata.")
        occupancy = st.slider("Animals currently in care", 0.0, 3000.0,
                              float(bundle.occupancy_default), step=10.0)
        intakes_7d = st.slider("Intakes over the previous 7 days", 0.0, 800.0,
                              float(bundle.intakes_7d_default), step=5.0)

row = make_row("profile", occupancy, intakes_7d)
_inference_started = perf_counter()
prediction = predict_intake(bundle, row).probability_by_class
_inference_seconds = perf_counter() - _inference_started

with output_col:
    st.markdown("#### Predicted outcome probabilities")
    st.caption("The model updates automatically as you change the profile.")
    st.metric("Highest probability · " + str(prediction.index[0]),
              f"{prediction.iloc[0]:.1%}")
    _chart_started = perf_counter()
    chart(prediction)
    _chart_seconds = perf_counter() - _chart_started
    st.caption("These are estimated probabilities, not certainties. "
               "'Other' pools several rare outcomes, not only euthanasia.")

st.divider()
with st.expander("Compare two profiles (optional)", expanded=False):
    st.write("Change one characteristic to see how the model responds. "
             "Differences show associative model behavior, **not causal effects**.")
    compare_enabled = st.toggle("Enable side-by-side comparison", value=False)
    if compare_enabled:
        if "comparison_initialized" not in st.session_state:
            for field in PRESETS["Stray dog"]:
                st.session_state[f"comparison_{field}"] = st.session_state[f"profile_{field}"]
            st.session_state["comparison_initialized"] = True
        with st.container(border=True):
            st.markdown("**Comparison profile**")
            profile_editor("comparison")
        other = predict_intake(bundle, make_row("comparison", occupancy, intakes_7d)).probability_by_class
        compare_chart(prediction, other)
        deltas = pd.DataFrame({
            "Outcome": list(prediction.index),
            "Original (%)": [round(100 * float(prediction[k]), 2) for k in prediction.index],
            "Comparison (%)": [round(100 * float(other[k]), 2) for k in prediction.index],
            "Change (percentage points)": [
                round(100 * (float(other[k]) - float(prediction[k])), 2) for k in prediction.index
            ],
        })
        st.dataframe(deltas, hide_index=True, width="stretch",
                     column_config={
                         "Original (%)": st.column_config.NumberColumn(format="%.1f"),
                         "Comparison (%)": st.column_config.NumberColumn(format="%.1f"),
                         "Change (percentage points)": st.column_config.NumberColumn(format="%+.2f"),
                     })
        st.download_button("Download comparison CSV", deltas.to_csv(index=False),
                           file_name="shelter_profile_comparison.csv", mime="text/csv")
    else:
        st.caption("Enable comparison to view two prediction distributions together.")

with st.expander("Model methodology and limitations"):
    st.markdown(f"""
**What is predicted?** Adoption, Return to Owner, Transfer, and Other, using
intake-time characteristics and historical shelter context.

**Evaluation:** Held-out animal-level macro-F1 of **{bundle.test_macro_f1:.3f}**.
Macro-F1 averages F1 across classes and is more informative than accuracy for
imbalanced outcomes. The model was also assessed on later Austin records and Sonoma data
in the accompanying report.

**Important caveats**
- *Other* combines euthanasia, died, disposal, missing, and other rare outcomes.
  It **must not** be read as a euthanasia probability.
- Historical records may reflect missing information and past institutional practices.
- Prediction differences between hypothetical profiles do not establish causality.
- The classifier is not calibrated or validated for real-time shelter decisions.
""")
st.caption("Source: What Intake Records Actually Tell You · Problem 1 · Frozen model artifact")

if _initial_session:
    _logger.info(
        "shelter_demo initial_session complete total_s=%.3f bundle_s=%.3f "
        "inference_s=%.3f chart_s=%.3f",
        perf_counter() - _run_started, _bundle_seconds, _inference_seconds, _chart_seconds,
    )
    st.session_state["_initial_timing_logged"] = True
