import streamlit as st

# Page title and introduction
st.title("Adult Census Income Prediction Project")
st.write("This project is focused on predicting whether an individual's income exceeds $50K/year based on census data.")

# Link to Colab code
st.markdown(
    """
    For the complete code, you can view and run the project on Google Colab:
    [Colab Project Link](https://colab.research.google.com/drive/1h5S9m2OWJH-RvbzotX0F9QKgI39ANeYk#scrollTo=70V_b5Aaztgt)
    """,
    unsafe_allow_html=True,
)

# Project documentation
st.subheader("Project Workflow")
st.write("""
1. **Data Preprocessing**:
   - **Label Encoding**: Converts categorical features into numeric labels to make them compatible with machine learning algorithms.
   - **Train-Test Split**: Splits the data into training and testing sets.
   - **Standardization**: Scales the features to have zero mean and unit variance using `StandardScaler`.

2. **Handling Imbalanced Data**:
   - **SMOTE (Synthetic Minority Over-sampling Technique)**: Balances the data by oversampling the minority class.
   - **Random Under-sampling**: Optionally reduces the majority class for balance, reducing training time with large datasets.

3. **Model Selection**:
    - The following models were considered for training:
     - **K-Nearest Neighbors (KNN)**: A simple, instance-based classification algorithm.
     - **Support Vector Classifier (SVC)**: A powerful classification model that finds an optimal hyperplane.
     - **Gaussian Naive Bayes (GNB)**: A probabilistic classifier that assumes feature independence.
     - **Decision Tree Classifier**: A tree-based model that makes decisions based on feature splits.
     - **Random Forest Classifier (RFC)**: An ensemble of decision trees for improved accuracy.
     - **AdaBoost Classifier**: A boosting algorithm that combines weak learners for better predictions.
     - **Gradient Boosting Classifier (GBC)**: Uses boosting for strong performance on complex datasets.
     - **XGBoost Classifier**: An efficient, scalable implementation of gradient boosting.

4. **Hyperparameter Tuning**:
   - **GridSearchCV** and **RandomizedSearchCV** were used to find the best hyperparameters for each model.

5. **Model Evaluation**:
    - **Classification Report**: Provides precision, recall, and F1-score for each class.
    - **Confusion Matrix**: Shows the true positives, true negatives, false positives, and false negatives.
    - **ROC Curve and AUC Score**: Plots the true positive rate against the false positive rate, assessing model performance.

6. **Saving the Model**:
    - Models and preprocessing objects were saved using `pickle` for easy reuse.
""")

st.subheader("Libraries Used")
st.write("""
- **Pandas**: For data manipulation and analysis.
- **Scikit-Learn**: Provides tools for preprocessing, model selection, and evaluation.
- **XGBoost**: Implements the gradient boosting algorithm.
- **Imbalanced-Learn**: Used for handling imbalanced datasets.
- **Warnings Filter**: Suppresses unnecessary warnings.
""")
