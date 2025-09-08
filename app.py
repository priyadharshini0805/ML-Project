### app.py ###
import streamlit as st
from tmodel import train_model
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def main():
    st.title("Enhanced Machine Learning Model Trainer")
    st.write("Upload your dataset, configure parameters, and visualize results.")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("File Uploaded Successfully!")
        st.write("### Dataset Overview")
        st.write(data.head())
        st.write("**Shape:**", data.shape)

        target_column = st.selectbox("Select Target Column", data.columns)
        features = data.drop(columns=[target_column])
        target = data[target_column]

        st.write("### Class Distribution")
        st.bar_chart(target.value_counts())

        # Handle Imbalance
        if st.checkbox("Balance Dataset"):
            from sklearn.utils import resample
            combined = pd.concat([features, target], axis=1)
            majority_class = target.value_counts().idxmax()
            balanced = combined[combined[target_column] == majority_class]

            for cls in target.value_counts().index:
                if cls != majority_class:
                    minority_data = combined[combined[target_column] == cls]
                    upsampled = resample(
                        minority_data, 
                        replace=True, 
                        n_samples=target.value_counts()[majority_class], 
                        random_state=42
                    )
                    balanced = pd.concat([balanced, upsampled])

            balanced = balanced.sample(frac=1).reset_index(drop=True)
            features = balanced.drop(columns=[target_column])
            target = balanced[target_column]
            st.write("Balanced Dataset Distribution")
            st.bar_chart(target.value_counts())

        # PCA Visualization
        if st.checkbox("Plot Dataset with PCA"):
            st.write("### PCA Projection")
            pca = PCA(2)
            X_projected = pca.fit_transform(features)

            x1 = X_projected[:, 0]
            x2 = X_projected[:, 1]

            fig = plt.figure()
            plt.scatter(x1, x2, c=target, alpha=0.8, cmap='viridis')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar()
            st.pyplot(fig)

        # Correlation Heatmap
        if st.checkbox("Show Correlation Heatmap"):
            st.write("### Correlation Heatmap")
            correlation_matrix = features.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # Model selection and training
    trained_models = []
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = []

    model_name = st.sidebar.selectbox("Select Model to Train", ["KNN", "SVM", "RF", "DT", "XTree"])

    # Hyperparameter selection
    params = {}
    if model_name == "KNN":
        params["n_neighbors"] = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5)
    elif model_name == "SVM":
        params["C"] = st.sidebar.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
        params["kernel"] = st.sidebar.selectbox("Kernel Type", ["linear", "rbf", "poly", "sigmoid"])
    elif model_name in ["RF", "XTree"]:
        params["n_estimators"] = st.sidebar.slider("Number of Estimators (Trees)", 10, 200, 100)
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 30, 10)
    elif model_name == "DT":
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 30, 10)

    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
    output_folder = st.sidebar.text_input("Output Folder Path", "model_output")
    epochs = st.sidebar.slider("Number of Epochs", 1, 10, 5)

    # Train model button
    if st.button("Train Model"):
        if uploaded_file is not None:
            results = train_model(features, target, model_name, params, test_size, random_state, output_folder, epochs)
            st.success(f"{model_name} Model Training Complete!")
            st.write("**Accuracy:**", results["accuracy"])
            st.write("**Precision:**", results["precision"])
            st.write("**Recall:**", results["recall"])
            st.write("**F1 Score:**", results["f1_score"])
            st.write(f"Model saved at: {results['model_path']}")

            # Confusion Matrix
            st.write("### Confusion Matrix")
            st.image(Image.open(results["conf_matrix_path"]), caption=f"{model_name} Confusion Matrix", use_column_width=True)

            # Train vs Validation Loss
            st.write("### Train vs Validation Loss")
            st.image(Image.open(results["train_val_curve_path"]), caption=f"{model_name} Loss Curve", use_column_width=True)

            # Save results for comparison
            results["model_name"] = model_name
            st.session_state.trained_models.append(results)

        else:
            st.error("Please upload a CSV file to proceed.")

    # Compare trained models
    if st.session_state.trained_models:
        if st.checkbox("Show Overall Model Comparison"):
            st.write("### Overall Model Comparison")
            comparison_df = pd.DataFrame(st.session_state.trained_models)
            st.dataframe(comparison_df)

            # Highlight best model based on accuracy
            best_model = comparison_df.loc[comparison_df['accuracy'].idxmax()]
            st.write("### Best Model")
            st.write(best_model)

            # Feature Importance
            if "feature_importances" in best_model and best_model["feature_importances"] is not None:
                st.write("### Feature Importance for Best Model")
                feature_importance = pd.DataFrame({
                    "Feature": features.columns,
                    "Importance": best_model["feature_importances"]
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(feature_importance.set_index("Feature"))

if __name__ == "__main__":
    main()