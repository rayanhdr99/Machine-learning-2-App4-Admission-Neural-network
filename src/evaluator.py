# this file evaluates the trained neural network and returns performance metrics
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)


# get predictions and check accuracy, confusion matrix, and classification report
def evaluate_model(model, X, y, dataset_name: str = "Test") -> dict:
    try:
        y_pred = model.predict(X)  # get predictions from the model
        acc = accuracy_score(y, y_pred)  # calculate overall accuracy
        cm = confusion_matrix(y, y_pred)  # build the confusion matrix (TP, FP, TN, FN)
        report = classification_report(y, y_pred, output_dict=True)  # detailed precision/recall/f1
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", dataset_name, e)
        raise
    logger.info("%s accuracy: %.4f", dataset_name, acc)
    # return everything in a dictionary so the app can display it
    return {"accuracy": acc, "confusion_matrix": cm, "report": report, "predictions": y_pred}
