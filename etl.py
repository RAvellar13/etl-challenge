"""
ETL pipeline for the Python Data Engineer code challenge.

Extracts candidate data from a CSV file, cleans and transforms it with
pandas, and loads it into a normalized SQLite relational database
(dimension tables for country / technology / seniority + a candidates
fact table). All downstream reporting/visualization must read from the
database, never from the CSV, per the challenge requirements.

Usage:
    python etl.py --csv data/candidates.csv --db db/candidates.db
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("etl")

# A candidate is considered HIRED when both the code challenge score and
# the technical interview score are 7 or higher (10-point scale).
HIRE_THRESHOLD = 7

EXPECTED_COLUMNS = {
    "First Name",
    "Last Name",
    "Email",
    "Application Date",
    "Country",
    "YOE",
    "Seniority",
    "Technology",
    "Code Challenge Score",
    "Technical Interview Score",
}


def extract(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame."""
    logger.info("Extracting data from %s", csv_path)
    df = pd.read_csv(csv_path, sep=";")

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    logger.info("Extracted %d rows", len(df))
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, validate and enrich the raw candidate data."""
    logger.info("Transforming data")
    df = df.copy()

    # --- Normalize text fields -------------------------------------------------
    text_cols = [
        "First Name",
        "Last Name",
        "Email",
        "Country",
        "Seniority",
        "Technology",
    ]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    df["Email"] = df["Email"].str.lower()

    # --- Parse dates -------------------------------------------------------
    df["Application Date"] = pd.to_datetime(df["Application Date"], errors="coerce")
    bad_dates = df["Application Date"].isna().sum()
    if bad_dates:
        logger.warning("Dropping %d rows with unparseable Application Date", bad_dates)
        df = df.dropna(subset=["Application Date"])

    # --- Validate numeric fields --------------------------------------------
    numeric_cols = ["YOE", "Code Challenge Score", "Technical Interview Score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=numeric_cols)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with invalid numeric values", dropped)

    # Scores are documented as a 0-10 scale; clip any out-of-range noise.
    for col in ("Code Challenge Score", "Technical Interview Score"):
        df[col] = df[col].clip(lower=0, upper=10)
    df["YOE"] = df["YOE"].clip(lower=0)

    df["YOE"] = df["YOE"].astype(int)
    df["Code Challenge Score"] = df["Code Challenge Score"].astype(int)
    df["Technical Interview Score"] = df["Technical Interview Score"].astype(int)

    # --- Remove exact duplicate rows ----------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    deduped = before - len(df)
    if deduped:
        logger.info("Removed %d exact duplicate rows", deduped)

    # --- Derived columns -----------------------------------------------------
    df["Application Year"] = df["Application Date"].dt.year
    df["Is Hired"] = (
        (df["Code Challenge Score"] >= HIRE_THRESHOLD)
        & (df["Technical Interview Score"] >= HIRE_THRESHOLD)
    ).astype(int)

    df = df.reset_index(drop=True)
    logger.info(
        "Transform complete: %d clean rows, %d hires (%.1f%%)",
        len(df),
        df["Is Hired"].sum(),
        100 * df["Is Hired"].mean(),
    )
    return df


DDL = """
CREATE TABLE IF NOT EXISTS dim_country (
    country_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    country_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_technology (
    technology_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    technology_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_seniority (
    seniority_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    seniority_name TEXT NOT NULL UNIQUE,
    seniority_rank INTEGER
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name                 TEXT NOT NULL,
    last_name                  TEXT NOT NULL,
    email                      TEXT NOT NULL,
    application_date           TEXT NOT NULL,
    application_year           INTEGER NOT NULL,
    yoe                        INTEGER NOT NULL,
    country_id                 INTEGER NOT NULL REFERENCES dim_country(country_id),
    technology_id              INTEGER NOT NULL REFERENCES dim_technology(technology_id),
    seniority_id               INTEGER NOT NULL REFERENCES dim_seniority(seniority_id),
    code_challenge_score       INTEGER NOT NULL,
    technical_interview_score  INTEGER NOT NULL,
    is_hired                   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidates_country ON candidates(country_id);
CREATE INDEX IF NOT EXISTS idx_candidates_technology ON candidates(technology_id);
CREATE INDEX IF NOT EXISTS idx_candidates_seniority ON candidates(seniority_id);
CREATE INDEX IF NOT EXISTS idx_candidates_year ON candidates(application_year);
CREATE INDEX IF NOT EXISTS idx_candidates_hired ON candidates(is_hired);
"""

# Fixed seniority ordering, used only for nicer chart ordering downstream.
SENIORITY_ORDER = [
    "Trainee",
    "Intern",
    "Junior",
    "Mid-Level",
    "Senior",
    "Lead",
    "Architect",
]


def load(df: pd.DataFrame, db_path: Path) -> None:
    """Create the schema from scratch and load the transformed data."""
    logger.info("Loading data into %s", db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # The ETL always does a full refresh, so start from a clean file rather
    # than relying on `CREATE TABLE IF NOT EXISTS`: that guard only skips
    # creation when a table of that name already exists, it does NOT check
    # whether its columns match the current schema. If a previous run (or
    # an older version of this script) left behind a db with a different
    # dim_seniority/candidates layout, IF NOT EXISTS would silently keep
    # the stale schema and the INSERTs below would fail with errors like
    # "table dim_seniority has no column named seniority_rank". Deleting
    # the file first makes every run schema-safe regardless of history.
    if db_path.exists():
        logger.info("Removing existing database at %s for a clean rebuild", db_path)
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(DDL)

        for name in sorted(df["Country"].unique()):
            conn.execute(
                "INSERT OR IGNORE INTO dim_country (country_name) VALUES (?)", (name,)
            )

        for name in sorted(df["Technology"].unique()):
            conn.execute(
                "INSERT OR IGNORE INTO dim_technology (technology_name) VALUES (?)",
                (name,),
            )

        seniorities = sorted(
            df["Seniority"].unique(),
            key=lambda s: SENIORITY_ORDER.index(s) if s in SENIORITY_ORDER else 99,
        )
        for rank, name in enumerate(seniorities):
            conn.execute(
                "INSERT OR IGNORE INTO dim_seniority (seniority_name, seniority_rank) "
                "VALUES (?, ?)",
                (name, rank),
            )
        conn.commit()

        country_map = dict(conn.execute("SELECT country_name, country_id FROM dim_country"))
        tech_map = dict(conn.execute("SELECT technology_name, technology_id FROM dim_technology"))
        seniority_map = dict(conn.execute("SELECT seniority_name, seniority_id FROM dim_seniority"))

        fact = pd.DataFrame(
            {
                "first_name": df["First Name"],
                "last_name": df["Last Name"],
                "email": df["Email"],
                "application_date": df["Application Date"].dt.strftime("%Y-%m-%d"),
                "application_year": df["Application Year"],
                "yoe": df["YOE"],
                "country_id": df["Country"].map(country_map),
                "technology_id": df["Technology"].map(tech_map),
                "seniority_id": df["Seniority"].map(seniority_map),
                "code_challenge_score": df["Code Challenge Score"],
                "technical_interview_score": df["Technical Interview Score"],
                "is_hired": df["Is Hired"],
            }
        )

        fact.to_sql("candidates", conn, if_exists="append", index=False)
        conn.commit()

        total = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        hired = conn.execute("SELECT COUNT(*) FROM candidates WHERE is_hired = 1").fetchone()[0]
        logger.info("Loaded %d candidate rows into the database (%d hired)", total, hired)
    finally:
        conn.close()


def run(csv_path: Path, db_path: Path) -> None:
    df = extract(csv_path)
    df = transform(df)
    load(df, db_path)
    logger.info("ETL finished successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("data/candidates.csv"),
        help="Path to the input CSV file (default: data/candidates.csv)",
    )
    parser.add_argument(
        "--db", type=Path, default=Path("db/candidates.db"),
        help="Path to the output SQLite database (default: db/candidates.db)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.csv, args.db)
