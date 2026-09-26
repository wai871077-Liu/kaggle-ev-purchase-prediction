from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SUBMISSION_DIR = ROOT / "submissions"

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DIR / "sample_submission.csv"
ORIGINAL_PATH = RAW_DIR / "original.csv"

TARGET = "Will_Buy_EV"
ID_COLUMN = "id"
SEED = 42
N_SPLITS = 5

NUMERIC_COLUMNS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_ENCODING_COLUMNS = ["Annual_Income_USD", "Daily_Commute_km"]

EXPECTED_TRAIN_ROWS = 668_665
EXPECTED_TEST_ROWS = 286_571


def ensure_output_directories() -> None:
    for directory in (PROCESSED_DIR, REPORT_DIR, FIGURE_DIR, SUBMISSION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
