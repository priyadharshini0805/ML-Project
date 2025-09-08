# Machine Learning Model Trainer with Streamlit

An interactive Streamlit application that enables users to upload datasets, train machine learning models, and analyze their performance. The app supports hyperparameter tuning, visualizations, and comparison of multiple models.

---

## Features

- **Dataset Upload and Overview**:
  - Upload `.csv` files and view dataset shape, class distribution, and sample data.
  - Automatically balance the dataset using upsampling if necessary.

- **Data Visualization**:
  - **Correlation Heatmap**: Explore relationships between features.
  - **PCA Plot**: Visualize data projected onto the first two principal components.

- **Model Training and Evaluation**:
  - Train individual models with dynamic hyperparameter tuning.
  - Supported Models:
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Random Forest (RF)
    - Decision Tree (DT)
    - Extra Trees (XTree)
  - Analyze performance metrics (Accuracy, Precision, Recall, F1-score).
  - View confusion matrix and training vs. validation loss curves.
  - Feature importance plots for applicable models.

- **Overall Model Comparison**:
  - Compare metrics across all trained models.
  - Feature importance plots for easy understanding of model behavior.

---

## Installation

### Clone the repository:
 ```bash 
git clone https://github.com/priya08dharshini/ML-projects.git
 ``` 
### Navigate to the project directory:
```bash
cd ML-projects
```
### Install dependencies:
  ```bash
  pip install -r requirements.txt
```
### Run the Streamlit application:
  ```bash
  streamlit run app.py
```

# Useage instruction
### 1. Upload Dataset
- Use the Upload CSV File button to upload your dataset.
- Ensure the target variable is the last column in your dataset.
### 2. Analyze and Preprocess
- Check the dataset overview for shape and sample data.
- Use the Balance Dataset checkbox if your dataset is imbalanced.
### 3. Visualize Data
- Enable the PCA Plot to observe feature separability.
- Use the Correlation Heatmap to analyze relationships between features.
### 4. Train Models
- Select a model from the sidebar.
- Configure hyperparameters dynamically.
- Click the Train Model button to train the selected model.
- View individual model performance, confusion matrix, and feature importance.
### 5. Compare Models
- After training multiple models, enable the Show Overall Model Comparison checkbox to compare performance metrics and feature importance for all trained models.

# Screenshots/ Videos

# File Structure
```bash
ML-projects/
├── app.py              # Main Streamlit application
├── tmodel.py           # Model training and evaluation logic
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── screenshots/        # Folder containing screenshots (optional)
└── uploded_file.csv/   # Example datasets 
```

# Technologies Used
- Programming Language: Python
- Framework: Streamlit
### Libraries:
- Pandas: For data manipulation and analysis.
- Scikit-learn: Machine learning algorithms and evaluation metrics.
- Matplotlib/Seaborn: For data visualization.

# Future Enhancements
- Add support for more models (e.g., XGBoost, Gradient Boosting).
- Integrate GridSearchCV for automatic hyperparameter optimization.
- Provide an option to download trained models in .pkl format.
- Enhance visualizations with more customization options.

