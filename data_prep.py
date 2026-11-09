"""Load and merge the Kaggle 'Formula 1 World Championship' CSV files.

Expected files inside the data/ folder:
    results.csv, races.csv, drivers.csv, circuits.csv, constructors.csv
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MISSING = ["\\N"]  # this dataset marks missing values as \N


def load_data() -> pd.DataFrame:
    """Return one row per driver per race, with names and helper columns."""
    results = pd.read_csv(DATA_DIR / "results.csv", na_values=MISSING)
    races = pd.read_csv(DATA_DIR / "races.csv", na_values=MISSING)
    drivers = pd.read_csv(DATA_DIR / "drivers.csv", na_values=MISSING)
    circuits = pd.read_csv(DATA_DIR / "circuits.csv", na_values=MISSING)
    constructors = pd.read_csv(DATA_DIR / "constructors.csv", na_values=MISSING)

    # Keep only the columns we need and give them clear names
    results = results[["raceId", "driverId", "constructorId", "grid", "positionOrder", "points"]]
    races = races[["raceId", "year", "round", "circuitId", "name"]].rename(
        columns={"name": "race_name"}
    )
    circuits = circuits[["circuitId", "name", "country"]].rename(
        columns={"name": "circuit_name"}
    )
    drivers = drivers.assign(driver=drivers["forename"] + " " + drivers["surname"])[
        ["driverId", "driver"]
    ]
    constructors = constructors[["constructorId", "name"]].rename(
        columns={"name": "constructor"}
    )

    # Merge everything into a single table
    df = (
        results.merge(races, on="raceId")
        .merge(circuits, on="circuitId")
        .merge(drivers, on="driverId")
        .merge(constructors, on="constructorId")
    )

    # Cleaning: grid == 0 means the driver started from the pit lane,
    # so there is no real grid slot to analyse.
    df = df.dropna(subset=["grid", "positionOrder"])
    df = df[df["grid"] > 0].copy()
    df["grid"] = df["grid"].astype(int)
    df["finish"] = df["positionOrder"].astype(int)

    # Helper columns
    df["positions_gained"] = df["grid"] - df["finish"]  # positive = moved forward
    df["won"] = (df["finish"] == 1).astype(int)
    df["podium"] = (df["finish"] <= 3).astype(int)
    df["pole"] = (df["grid"] == 1).astype(int)
    return add_team_form(df)


def add_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Add prev_pts: the team's average points per entry in the PREVIOUS season.

    Team strength changes every year, so last season's form is a fair proxy
    for how strong the car is. Using the previous season (not the current one)
    avoids leaking the race result into the feature.
    Teams with no previous season (new teams) get 0.
    """
    form = df.groupby(["constructor", "year"])["points"].mean().rename("prev_pts").reset_index()
    form["year"] += 1  # last season's value becomes this season's feature
    df = df.merge(form, on=["constructor", "year"], how="left")
    df["prev_pts"] = df["prev_pts"].fillna(0)
    return df


def latest_team_form(df: pd.DataFrame) -> pd.Series:
    """Average points per entry in the most recent season, for each team."""
    last = df[df["year"] == df["year"].max()]
    return last.groupby("constructor")["points"].mean().sort_values(ascending=False).round(2)


def grid_summary(df: pd.DataFrame, max_grid: int = 20) -> pd.DataFrame:
    """Average finish, win rate and podium rate for each starting position."""
    out = (
        df[df["grid"] <= max_grid]
        .groupby("grid")
        .agg(
            avg_finish=("finish", "mean"),
            win_rate=("won", "mean"),
            podium_rate=("podium", "mean"),
            samples=("finish", "size"),
        )
        .round(3)
    )
    return out


def circuit_summary(df: pd.DataFrame, min_races: int = 5) -> pd.DataFrame:
    """Per-circuit stats showing how much grid position matters there.

    grid_finish_corr: Spearman correlation between grid and finish.
        Closer to 1 = the grid order largely decides the result (hard to overtake).
    pole_win_rate: share of races won from pole position.
    """
    rows = []
    for name, g in df.groupby("circuit_name"):
        n_races = g["raceId"].nunique()
        if n_races < min_races:
            continue
        poles = g[g["pole"] == 1]
        rows.append(
            {
                "circuit_name": name,
                "country": g["country"].iloc[0],
                "races": n_races,
                "grid_finish_corr": g["grid"].corr(g["finish"], method="spearman"),
                "pole_win_rate": poles["won"].mean() if len(poles) else float("nan"),
                "avg_abs_position_change": g["positions_gained"].abs().mean(),
            }
        )
    out = pd.DataFrame(rows).set_index("circuit_name").round(3)
    return out.sort_values("grid_finish_corr", ascending=False)


if __name__ == "__main__":
    data = load_data()
    print(data.head())
    print(f"\nRows: {len(data)} | Seasons: {data['year'].min()}-{data['year'].max()}")
