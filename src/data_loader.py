# this file handles loading and validating the UCLA admissions dataset
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# these are the columns we expect in the csv file
REQUIRED_COLUMNS = [
    "Serial_No", "GRE_Score", "TOEFL_Score", "University_Rating",
    "SOP", "LOR", "CGPA", "Research", "Admit_Chance",
]


# load the admissions csv and validate the columns
def load_data(filepath: str) -> pd.DataFrame:
    logger.info("Loading data from: %s", filepath)
    try:
        df = pd.read_csv(filepath)  # read the csv into a pandas dataframe
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Data file not found: {filepath}") from e
    except Exception as e:
        logger.error("Failed to read CSV: %s", e)
        raise

    # make sure the dataframe isn't empty
    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    # check that all required columns are present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    logger.info("Data loaded successfully. Shape: %s", df.shape)
    return df
