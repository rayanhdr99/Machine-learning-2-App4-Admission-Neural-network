# this file handles all the data preprocessing - dropping columns, encoding, scaling, splitting
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# prepare the features by dropping the ID column, converting target to binary, and one-hot encoding
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preparing features.")
    df = df.copy()  # work on a copy so we don't modify the original

    # drop Serial_No since it's just an ID, not a feature
    df = df.drop("Serial_No", axis=1)

    # convert admission chance to binary - 1 if >= 0.8, else 0
    df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

    # cast these to object type so pandas treats them as categorical for one-hot encoding
    df["University_Rating"] = df["University_Rating"].astype("object")
    df["Research"] = df["Research"].astype("object")

    # one-hot encode university rating and research columns
    df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)

    logger.info("Feature preparation complete. Columns: %s", list(df.columns))
    return df


# split the data into train/test sets and scale features to 0-1 range
def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 123):
    logger.info("Splitting and scaling data.")

    # separate features (X) from the target variable (y)
    X = df.drop("Admit_Chance", axis=1)
    y = df["Admit_Chance"]
    feature_columns = list(X.columns)  # save column names for later use in prediction

    # do an 80/20 train-test split, stratified so both sets have similar class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # scale features to 0-1 range so the neural network trains better
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on training data only
    X_test_scaled = scaler.transform(X_test)  # transform test data using same scaler

    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns
