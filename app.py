# this is the main streamlit app for the UCLA admission chance predictor
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# make sure python can find our src modules in the same directory
sys.path.insert(0, os.path.dirname(__file__))

# import our custom modules for loading data, preprocessing, training, and evaluation
from src.data_loader import load_data
from src.preprocessor import prepare_features, split_and_scale
from src.model import train_neural_network
from src.evaluator import evaluate_model

# set up logging so we can track what's happening in the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# path to the admissions dataset csv file
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Admission.csv")

# configure the streamlit page with a title, icon, and wide layout
st.set_page_config(page_title="UCLA Admission Predictor", page_icon="🎓", layout="wide")


# cache the data so we don't reload the csv every time the app refreshes
@st.cache_data
def get_data():
    return load_data(DATA_PATH)


# cache the trained model and all related objects so we only train once
@st.cache_resource
def get_model():
    df = get_data()  # load raw data
    df_processed = prepare_features(df)  # preprocess features (drop id, binarize target, one-hot encode)
    X_train_s, X_test_s, y_train, y_test, scaler, feature_cols = split_and_scale(df_processed)  # split and scale
    model = train_neural_network(X_train_s, y_train)  # train the neural network
    return model, X_train_s, X_test_s, y_train, y_test, scaler, feature_cols


def main():
    # app title and description at the top of the page
    st.title("🎓 UCLA Admission Chance Predictor")
    st.markdown("Predict a student's chance of admission to UCLA using a **Neural Network (MLP Classifier)**.")

    # sidebar navigation to switch between pages
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Overview", "Model Performance", "Predict Admission"])

    # load data and model; show an error if anything goes wrong
    try:
        df_raw = get_data()
        model, X_train_s, X_test_s, y_train, y_test, scaler, feature_cols = get_model()
    except Exception as e:
        st.error(f"Setup error: {e}")
        logger.error("App startup error: %s", e)
        return

    # ---------- PAGE 1: Dataset Overview ----------
    if page == "Dataset Overview":
        st.header("Dataset Overview")

        # count how many students would be admitted (admit chance >= 0.8)
        admitted = (df_raw["Admit_Chance"] >= 0.8).sum()

        # display key metrics in three columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", df_raw.shape[0])
        col2.metric("Would Be Admitted (>=80%)", f"{admitted} ({admitted/len(df_raw)*100:.0f}%)")
        col3.metric("Avg CGPA", f"{df_raw['CGPA'].mean():.2f}/10")

        # show first 10 rows of the raw dataset
        st.subheader("Sample Data")
        st.dataframe(df_raw.head(10), use_container_width=True)

        # show basic stats like mean, std, min, max for each column
        st.subheader("Statistical Summary")
        st.dataframe(df_raw.describe(), use_container_width=True)

        # plot a heatmap to see which features are most correlated
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_raw.drop("Serial_No", axis=1).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        plt.close(fig)

        # plot histograms for GRE, TOEFL, and CGPA to see their distributions
        st.subheader("Score Distributions")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, col, color in zip(axes, ["GRE_Score", "TOEFL_Score", "CGPA"], ["steelblue", "teal", "salmon"]):
            ax.hist(df_raw[col], bins=20, color=color, edgecolor="white")
            ax.set_title(f"{col} Distribution")
            ax.set_xlabel(col)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ---------- PAGE 2: Model Performance ----------
    elif page == "Model Performance":
        st.header("Neural Network Performance")

        # evaluate on both training and test sets to check for overfitting
        train_res = evaluate_model(model, X_train_s, y_train, "Train")
        test_res = evaluate_model(model, X_test_s, y_test, "Test")

        # show train vs test accuracy side by side
        col1, col2 = st.columns(2)
        col1.metric("Train Accuracy", f"{train_res['accuracy']*100:.2f}%")
        col2.metric("Test Accuracy", f"{test_res['accuracy']*100:.2f}%")

        # display the confusion matrix as a heatmap
        st.subheader("Confusion Matrix (Test Set)")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(test_res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Admitted", "Admitted"],
                    yticklabels=["Not Admitted", "Admitted"], ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        # plot the loss curve to see how the model improved during training
        st.subheader("Training Loss Curve")
        if hasattr(model, "loss_curve_"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(model.loss_curve_, color="steelblue", linewidth=2)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            ax.set_title("Neural Network Training Loss")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        # show precision, recall, f1-score breakdown
        st.subheader("Classification Report")
        report_df = pd.DataFrame(test_res["report"]).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        # display the model architecture details
        st.subheader("Model Architecture")
        st.info(
            f"**Hidden Layers:** 1 layer with 3 neurons  \n"
            f"**Activation:** tanh  \n"
            f"**Batch Size:** 50  \n"
            f"**Max Iterations:** 200  \n"
            f"**Input Features:** {len(feature_cols)}"
        )

    # ---------- PAGE 3: Predict Admission ----------
    elif page == "Predict Admission":
        st.header("Predict Admission Chance")
        st.markdown("Enter your academic profile to predict if you would be admitted to UCLA.")

        # input widgets split into two columns for a cleaner layout
        col1, col2 = st.columns(2)
        with col1:
            gre_score = st.slider("GRE Score (out of 340)", min_value=260, max_value=340, value=316)
            toefl_score = st.slider("TOEFL Score (out of 120)", min_value=92, max_value=120, value=107)
            university_rating = st.selectbox("Bachelor's University Rating (1-5)", [1, 2, 3, 4, 5], index=2)
            sop = st.slider("Statement of Purpose (SOP) Strength (out of 5)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
        with col2:
            lor = st.slider("Letter of Recommendation (LOR) Strength (out of 5)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
            cgpa = st.slider("CGPA (out of 10)", min_value=6.0, max_value=10.0, value=8.5, step=0.01)
            research = st.selectbox("Research Experience", [0, 1], format_func=lambda x: "Yes" if x else "No")

        # when the user clicks predict, build the input and run the model
        if st.button("Predict Admission", type="primary"):
            try:
                # build a dictionary with the numeric features
                input_dict = {
                    "GRE_Score": gre_score,
                    "TOEFL_Score": toefl_score,
                    "SOP": sop,
                    "LOR": lor,
                    "CGPA": cgpa,
                }
                # One-hot encode University_Rating (1-5)
                for i in range(1, 6):
                    input_dict[f"University_Rating_{i}"] = 1 if university_rating == i else 0
                # One-hot encode Research (0 or 1)
                input_dict["Research_0"] = 1 if research == 0 else 0
                input_dict["Research_1"] = 1 if research == 1 else 0

                # create a dataframe from the input and make sure columns match training data
                input_df = pd.DataFrame([input_dict])
                for col in feature_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_cols]

                # scale the input using the same scaler from training
                input_scaled = scaler.transform(input_df)

                # get prediction and probability from the model
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]

                # show result based on whether the model predicts admitted or not
                if prediction == 1:
                    st.success(f"Likely to be Admitted!  \nProbability: {proba[1]*100:.1f}%")
                else:
                    st.warning(f"Unlikely to be Admitted.  \nProbability of admission: {proba[1]*100:.1f}%  \nConsider improving your GRE, CGPA, or research experience.")

                # display admission vs rejection probability as metrics
                col1, col2 = st.columns(2)
                col1.metric("Admission Probability", f"{proba[1]*100:.1f}%")
                col2.metric("Rejection Probability", f"{proba[0]*100:.1f}%")

                logger.info("Prediction: %d  Proba: %.2f", prediction, proba[1])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.error("Prediction error: %s", e)


if __name__ == "__main__":
    main()
