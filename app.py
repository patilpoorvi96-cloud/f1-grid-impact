"""F1 Grid Impact Explorer - Streamlit app.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from data_prep import circuit_summary, grid_summary, latest_team_form, load_data
from model import (
    MODERN_ERA_START,
    TEST_FROM_YEAR,
    evaluate,
    predict_podium_probability,
    train_final_model,
)

st.set_page_config(page_title="F1 Grid Impact Explorer", page_icon="🏎️", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_model(_df: pd.DataFrame):
    return train_final_model(_df)


@st.cache_data
def get_metrics(_df: pd.DataFrame) -> dict:
    return evaluate(_df)


try:
    df = get_data()
except FileNotFoundError as err:
    st.error(
        f"Data file not found: {err.filename}\n\n"
        "Download the Kaggle dataset 'Formula 1 World Championship (1950 - 2020)' "
        "and put results.csv, races.csv, drivers.csv, circuits.csv and "
        "constructors.csv inside the data/ folder."
    )
    st.stop()

st.title("🏎️ F1 Grid Impact Explorer")
st.caption("How much does your starting position decide the race result?")

# ---- Sidebar filter -------------------------------------------------------
min_year, max_year = int(df["year"].min()), int(df["year"].max())
years = st.sidebar.slider("Seasons", min_year, max_year, (MODERN_ERA_START, max_year))
filtered = df[df["year"].between(*years)]
st.sidebar.write(f"{filtered['raceId'].nunique()} races | {len(filtered)} driver entries")

tab1, tab2, tab3 = st.tabs(["Grid vs. finish", "Circuit explorer", "Podium predictor"])

# ---- Tab 1: grid vs. finish ----------------------------------------------
with tab1:
    summary = grid_summary(filtered)
    st.subheader("Average finishing position by starting position")
    st.line_chart(summary["avg_finish"], x_label="Grid position", y_label="Average finish")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Win rate by grid slot")
        st.bar_chart(summary["win_rate"], x_label="Grid position", y_label="Win rate")
    with col2:
        st.subheader("Podium rate by grid slot")
        st.bar_chart(summary["podium_rate"], x_label="Grid position", y_label="Podium rate")

    pole_win = summary.loc[1, "win_rate"] if 1 in summary.index else float("nan")
    st.info(f"In the selected seasons, the driver on pole won **{pole_win:.0%}** of races.")
    with st.expander("See the data table"):
        st.dataframe(summary)

# ---- Tab 2: circuit explorer ---------------------------------------------
with tab2:
    min_races = st.slider("Minimum races at circuit", 1, 20, 3)
    circuits = circuit_summary(filtered, min_races=min_races)
    if circuits.empty:
        st.warning("No circuits match these filters. Try widening the seasons or lowering the minimum.")
    else:
        st.subheader("Where does the grid decide the race?")
        st.caption(
            "Grid-finish correlation near 1 means the starting order mostly decides the result "
            "(overtaking is hard). Lower means more shuffling."
        )
        st.bar_chart(circuits["grid_finish_corr"], horizontal=True, height=max(300, 22 * len(circuits)))

        st.subheader("Compare two circuits")
        names = list(circuits.index)
        c1, c2 = st.columns(2)
        a = c1.selectbox("Circuit A", names, index=0)
        b = c2.selectbox("Circuit B", names, index=len(names) - 1)
        st.dataframe(circuits.loc[[a, b]].drop(columns="country").T)

        for label, name in [("A", a), ("B", b)]:
            cs = grid_summary(filtered[filtered["circuit_name"] == name], max_grid=10)
            st.write(f"**Circuit {label}: {name}** - average finish by grid (top 10 starters)")
            st.line_chart(cs["avg_finish"], x_label="Grid position", y_label="Average finish")

# ---- Tab 3: predictor ----------------------------------------------------
with tab3:
    st.subheader("Estimate podium chance")
    model = get_model(df)
    form = latest_team_form(df)
    last_season = int(df["year"].max())

    c1, c2 = st.columns(2)
    grid = c1.slider("Grid position", 1, 20, 3)
    team = c2.selectbox(f"Team (form = average points per race in {last_season})", list(form.index))

    prob = predict_podium_probability(model, grid, float(form[team]))
    st.metric("Estimated podium probability", f"{prob:.0%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"{team} scored {form[team]:.1f} points per entry in {last_season}, which is used as its form.")

    st.subheader("How good is this model?")
    metrics = get_metrics(df)
    if metrics:
        st.write(
            f"Trained on {MODERN_ERA_START}-{TEST_FROM_YEAR - 1} ({metrics['train_rows']} rows) and "
            f"tested on {TEST_FROM_YEAR} onwards ({metrics['test_rows']} rows)."
        )
        st.table(pd.DataFrame({k: v for k, v in metrics.items() if isinstance(v, dict)}).T)
        st.caption(
            f"Always guessing 'no podium' would be right {metrics['baseline_accuracy_always_no_podium']:.0%} "
            "of the time, so ROC AUC is the more useful number here."
        )
    st.warning(
        "Limitations: the model ignores weather, tyre strategy, safety cars, reliability and "
        "driver skill. Treat the output as an estimate, not a prediction of any real race."
    )
