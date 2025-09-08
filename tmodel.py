import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

def train_model(X, y, model_name, params, test_size, random_state, output_folder, epochs=5):
    # Encode categorical features and target
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize model
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    elif model_name == "SVM":
        model = SVC(C=params["C"], kernel=params["kernel"], probability=True, random_state=random_state)
    elif model_name == "RF":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=random_state
        )
    elif model_name == "DT":
        model = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=random_state)
    elif model_name == "XTree":
        model = ExtraTreesClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=random_state
        )
    else:
        raise ValueError("Invalid model name")

    # Train and record losses
    train_losses, val_losses = [], []
    feature_importances = None
    for epoch in range(1, epochs + 1):
        if hasattr(model, "warm_start") and model.warm_start:
            model.set_params(n_estimators=epoch * params["n_estimators"] // epochs)
        model.fit(X_train, y_train)

        train_loss = log_loss(y_train, model.predict_proba(X_train)) if hasattr(model, "predict_proba") else 0
        val_loss = log_loss(y_test, model.predict_proba(X_test)) if hasattr(model, "predict_proba") else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }

    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_

    # Save model
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model_path = os.path.join(output_folder, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)

    # Plot Train vs Validation Loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.title("Train vs Validation Loss")
    loss_curve_path = os.path.join(output_folder, f"{model_name}_loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_folder, f"{model_name}_conf_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    return {
        **metrics,
        "model_path": model_path,
        "train_val_curve_path": loss_curve_path,
        "conf_matrix_path": cm_path,
        "feature_importances": feature_importances
    }
