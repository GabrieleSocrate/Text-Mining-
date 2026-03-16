from pathlib import Path
import ast
import pandas as pd
from functools import lru_cache


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "dashboard"

FULL_DATASET_PATH = DATA_DIR / "full_dataset.csv"
BALANCED_DATASET_PATH = DATA_DIR / "balanced_dataset.csv"


def _parse_list_column(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if item not in [None, "", "None"]]

    value = str(value).strip()

    if value in ["", "[]", "[None]", "None", "nan"]:
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item not in [None, "", "None"]]
    except Exception:
        pass

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []

        parts = [item.strip() for item in inner.split(",")]
        return [
            item.strip("'").strip('"')
            for item in parts
            if item and item not in ["None", ""]
        ]

    return [value]


def _prepare_dataset(path: Path) -> pd.DataFrame:
    print(f"Loading dataset: {path.name}")

    df = pd.read_csv(path)

    if "date_publish" in df.columns:
        df["date_publish"] = pd.to_datetime(df["date_publish"], errors="coerce")

    list_columns = ["mentioned_companies", "related_companies", "industries", "sector_group"]

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_column)

    if "sentiment_label" in df.columns:
        df["sentiment_label"] = df["sentiment_label"].astype(str).str.lower().str.strip()

    print(f"Finished loading dataset: {path.name}")

    return df


# CACHE LAZY
@lru_cache(maxsize=2)
def _get_cached_dataset(dataset_name: str) -> pd.DataFrame:

    if dataset_name == "full":
        return _prepare_dataset(FULL_DATASET_PATH)

    if dataset_name == "balanced":
        return _prepare_dataset(BALANCED_DATASET_PATH)

    raise ValueError("dataset_name must be 'full' or 'balanced'")


def load_dataset(dataset_name: str) -> pd.DataFrame:
    # restituiamo una copia per evitare modifiche accidentali
    return _get_cached_dataset(dataset_name).copy()


def get_available_companies(df: pd.DataFrame) -> list[str]:

    if "mentioned_companies" not in df.columns:
        return []

    companies = set()

    for items in df["mentioned_companies"]:
        if isinstance(items, list):
            for company in items:
                company = str(company).strip()

                if company and company.lower() != "none":
                    companies.add(company)

    return sorted(companies)


def get_available_sectors(df: pd.DataFrame) -> list[str]:

    if "sector_group" not in df.columns:
        return []

    sectors = set()

    for items in df["sector_group"]:
        if isinstance(items, list):
            for sector in items:
                sector = str(sector).strip()

                if sector and sector.lower() != "none":
                    sectors.add(sector)

    return sorted(sectors)


def get_date_bounds(df: pd.DataFrame):

    if "date_publish" not in df.columns:
        return None, None

    valid_dates = df["date_publish"].dropna()

    if valid_dates.empty:
        return None, None

    return valid_dates.min().date(), valid_dates.max().date()


def get_dataset_metadata(dataset_name: str) -> dict:

    df = load_dataset(dataset_name)

    companies = get_available_companies(df)
    sectors = get_available_sectors(df)

    min_date, max_date = get_date_bounds(df)

    return {
        "companies": companies,
        "sectors": sectors,
        "min_date": min_date,
        "max_date": max_date,
    }


def filter_dataset(dataset_name: str, selected_companies=None, start_date=None, end_date=None) -> pd.DataFrame:

    df = load_dataset(dataset_name)

    if "date_publish" in df.columns:

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df = df[df["date_publish"] >= start_date]

        if end_date is not None:
            end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df["date_publish"] <= end_date]

    if selected_companies:

        selected_companies_set = {str(company).strip() for company in selected_companies}

        df = df[
            df["mentioned_companies"].apply(
                lambda companies: any(company in selected_companies_set for company in companies)
                if isinstance(companies, list) else False
            )
        ]

    return df.reset_index(drop=True)