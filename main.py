import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def load_and_prepare_data(path):
    df = pd.read_csv(path)

    # Create target
    conditions = [
        df['Target_Severity_Score'] <= 3,
        df['Target_Severity_Score'] <= 6,
        df['Target_Severity_Score'] > 6
    ]
    choices = [0, 1, 2]
    df['Severity_Class'] = np.select(conditions, choices)

    # Drop unnecessary columns
    df = df.drop(columns=['Patient_ID', 'Year', 'Target_Severity_Score'])

    X = df.drop('Severity_Class', axis=1)
    y = df['Severity_Class']
    return X, y


def build_pipeline():
    numeric_features = ['Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Treatment_Cost_USD', 'Survival_Years', 'Age']
    categorical_features = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, C=0.1, solver='liblinear'))
    ])

    return model_pipeline


def main():
    X, y = load_and_prepare_data('./global_cancer_patients_2015_2024.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline = build_pipeline()

    # Set or create experiment
    mlflow.set_experiment("Cancer_Severity_Experiment")

    # MLflow tracking
    with mlflow.start_run():
        mlflow.set_tag("project", "Cancer Severity Classification")
        mlflow.set_tag("developer", "Your Name")

        # Log model parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 0.1)
        mlflow.log_param("solver", "liblinear")

        # Cross-validation accuracy
        scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy')
        cv_accuracy = scores.mean()
        mlflow.log_metric("cv_accuracy", cv_accuracy)

        # Fit model and evaluate
        model_pipeline.fit(X_train, y_train)
        test_accuracy = model_pipeline.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Compute and log confusion matrix
        y_pred = model_pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)

        # Save the confusion matrix image
        os.makedirs("outputs", exist_ok=True)
        cm_path = "outputs/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # Log image to MLflow
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

        # Log the model itself
        mlflow.sklearn.log_model(model_pipeline, "model")


if __name__ == "__main__":
    main()
