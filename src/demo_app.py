"""Streamlit UI for the Problem 1 outcome-classification demonstration.

Run:
    streamlit run src/demo_app.py

Frozen-artifact loading and inference live in ``demo_service.py`` so this module
only collects intake-time inputs and renders classifier outputs.
"""

import calendar

import altair as alt
import pandas as pd
import streamlit as st

from demo_service import load_demo_model, predict_intake

st.set_page_config(
    page_title="Shelter Outcome Explorer",
    page_icon="🐾",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading the frozen demo model...")
def load_demo_bundle():
    return load_demo_model()


bundle = load_demo_bundle()

st.title("🐾 Shelter Outcome Explorer")
st.caption(
    "Explore how intake-time characteristics change the Austin-trained model's "
    "predicted outcome probabilities for hypothetical animal profiles."
)

with st.sidebar:
    st.header("About this demo")
    st.markdown(
        f"- **Task:** four-class outcome prediction\n"
        f"- **Test design:** animal-level 80/20 split\n"
        f"- **Held-out macro-F1:** {bundle.test_macro_f1:.3f}\n"
        f"- **External check:** Sonoma County, reported in the writeup"
    )

with st.expander("What this demo does — and does not do"):
    st.markdown(
        """
- Uses only features available at intake time.
- Shows the classifier's historical outcome probabilities for Austin-like inputs.
- Does **not** recommend treatment, placement, euthanasia, or any other operational action.
- Does **not** demonstrate the association analyses in Problems 2 and 3.
        """
    )

st.caption(
    "Change one field at a time to see how the predicted probabilities shift."
)

col1, col2 = st.columns(2)

with col1:
    animal_type = st.selectbox("Species", ["Dog", "Cat", "Other"])
    sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])
    fixed = st.selectbox("Spay/neuter status", ["Intact", "Altered", "Unknown"])
    age_years = st.slider("Age (years)", 0.0, 20.0, 2.0, 0.5)

with col2:
    intake_type = st.selectbox(
        "Intake type",
        ["Stray", "Owner Surrender", "Public Assist", "Abandoned", "Other"],
    )
    condition_group = st.selectbox(
        "Intake condition",
        ["Normal", "Injured", "Medical", "Behavior/Other"],
    )
    is_black = st.checkbox("Black coat")
    month = st.selectbox(
        "Intake month",
        list(range(1, 13)),
        index=pd.Timestamp.now().month - 1,
        format_func=lambda m: calendar.month_name[m],
    )

with st.expander("Shelter context (defaults = representative Austin levels)"):
    occupancy = st.slider(
        "Current occupancy (animals in care)",
        0.0,
        3000.0,
        bundle.occupancy_default,
        10.0,
    )
    intakes_7d = st.slider(
        "Intakes in the last 7 days",
        0.0,
        800.0,
        bundle.intakes_7d_default,
        5.0,
    )

row = pd.DataFrame(
    [
        {
            "animal_type": animal_type,
            "sex": sex,
            "fixed": fixed,
            "intake_type": intake_type,
            "condition_group": condition_group,
            "is_black": is_black,
            "month": str(month),
            "age_years": age_years,
            "occupancy": occupancy,
            "intakes_7d": intakes_7d,
        }
    ]
)

prediction = predict_intake(bundle, row)
prob_by_class = prediction.probability_by_class

st.divider()

st.caption("Most likely outcome")
st.subheader(prob_by_class.index[0])
st.caption(f"{prob_by_class.iloc[0]:.0%} predicted probability")

st.subheader("Outcome probabilities")

probability_df = (
    prob_by_class.rename("probability")
    .rename_axis("outcome")
    .reset_index()
)

bars = (
    alt.Chart(probability_df)
    .mark_bar(color="#0072B2", cornerRadiusEnd=2)
    .encode(
        x=alt.X(
            "probability:Q",
            title="Predicted probability",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        y=alt.Y(
            "outcome:N",
            title=None,
            sort="-x",
        ),
        tooltip=[
            alt.Tooltip("outcome:N", title="Outcome"),
            alt.Tooltip(
                "probability:Q",
                title="Predicted probability",
                format=".1%",
            ),
        ],
    )
)

labels = bars.mark_text(
    align="left",
    baseline="middle",
    dx=4,
).encode(
    text=alt.Text("probability:Q", format=".1%")
)

probability_chart = (
    (bars + labels)
    .properties(height=190)
    .configure_view(stroke=None)
)

st.altair_chart(
    probability_chart,
    width="stretch",
    theme="streamlit",
)

st.caption(
    "Other is a pooled category containing Euthanasia, Died, Disposal, Missing, "
    "and other rare outcomes. It should not be interpreted as a euthanasia probability."
)

st.caption(
    "Predictions use intake-time information only. This is an illustrative model "
    "explorer, not an operational shelter decision tool. Historical predictions "
    "may reflect patterns and biases in past shelter practice."
)
