import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load scaler and models
ss = pickle.load(open('D:\luminar\project_1_ml\mlprj1\pages\model\scalar.pkl', 'rb'))
knn = pickle.load(open('D:\luminar\project_1_ml\mlprj1\pages\model\knn.pkl', 'rb'))
svc = pickle.load(open('D:\luminar\project_1_ml\mlprj1\pages\model\svc.pkl', 'rb'))
gnb = pickle.load(open("D:\luminar\project_1_ml\mlprj1\pages\model\gnb.pkl", "rb"))
griddt = pickle.load(open("D:\luminar\project_1_ml\mlprj1\pages\model\dtgrid.pkl", "rb"))
gridrfc = pickle.load(open("D:\luminar\project_1_ml\mlprj1\pages\model\_rfcgrid.pkl", "rb"))
ada = pickle.load(open("D:\luminar\project_1_ml\mlprj1\pages\model\_ada.pkl", "rb"))
gbc = pickle.load(open("D:\luminar\project_1_ml\mlprj1\pages\model\gbc.pkl", "rb"))
xgb = pickle.load(open("D:\luminar\project_1_ml\mlprj1\pages\model\_xgb.pkl", "rb"))

# Encoding function
def encode_input(data, encoder):
    return encoder.fit_transform([data])[0]

# Label Encoders for categorical variables
workclass_encoder = LabelEncoder()
education_encoder = LabelEncoder()
marital_status_encoder = LabelEncoder()
occupation_encoder = LabelEncoder()
relationship_encoder = LabelEncoder()
race_encoder = LabelEncoder()
sex_encoder = LabelEncoder()
native_country_encoder = LabelEncoder()

# Streamlit app UI
st.title("Income Prediction Based on Census Data")

with st.form(key='input_form'):
    age = st.slider("Age", 18, 120, 30)
    workclass = st.selectbox("Workclass", ['?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
                                           'Self-emp-not-inc', 'State-gov', 'Without-pay'])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1500000, 50000)
    education = st.selectbox("Education",
                             ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc',
                              'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school',
                              'Some-college'])
    education_num = st.slider("Education Number", 1, 16, 10)
    marital_status = st.selectbox("Marital Status",
                                  ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent',
                                   'Never-married', 'Separated', 'Widowed'])
    occupation = st.selectbox("Occupation", ['?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                                             'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
                                             'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
                                             'Sales', 'Tech-support', 'Transport-moving'])
    relationship = st.selectbox("Relationship",
                                ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'])
    race = st.selectbox("Race", ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'])
    sex = st.selectbox("Sex", ['Female', 'Male'])
    capital_gain = st.number_input("Capital Gain", 0, 99999, 5000)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 200)
    hours_per_week = st.slider("Hours per week", 1, 100, 40)
    native_country = st.selectbox("Native Country",
                                  ['?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
                                   'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala',
                                   'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran',
                                   'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua',
                                   'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal',
                                   'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinidad&Tobago',
                                   'United-States', 'Vietnam', 'Yugoslavia'])

    # Model selection dropdown
    model_choice = st.selectbox("Select a model",
                                ["KNN", "SVC", "GNB", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting",
                                 "XGBoost"])
    submit_button = st.form_submit_button(label="Submit")

# Process input data and make prediction
if submit_button:
    # Encode categorical values
    workclass_encoded = encode_input(workclass, workclass_encoder)
    education_encoded = encode_input(education, education_encoder)
    marital_status_encoded = encode_input(marital_status, marital_status_encoder)
    occupation_encoded = encode_input(occupation, occupation_encoder)
    relationship_encoded = encode_input(relationship, relationship_encoder)
    race_encoded = encode_input(race, race_encoder)
    sex_encoded = encode_input(sex, sex_encoder)
    native_country_encoded = encode_input(native_country, native_country_encoder)

    # Prepare the input data
    input_data = np.array([
        age,
        workclass_encoded,
        fnlwgt,
        education_encoded,
        education_num,
        marital_status_encoded,
        occupation_encoded,
        relationship_encoded,
        race_encoded,
        sex_encoded,
        capital_gain,
        capital_loss,
        hours_per_week,
        native_country_encoded
    ]).reshape(1, -1)

    # Scale the input data
    input_data_scaled = ss.transform(input_data)

    # Predict based on model choice
    if model_choice == "KNN":
        prediction = knn.predict(input_data_scaled)
        st.image("images\knn.png", caption="Confusion Matrix for KNN")
        st.image("images\knnroc.png", caption="ROC Curve for KNN")
    elif model_choice == "SVC":
        prediction = svc.predict(input_data_scaled)
        st.image("images\svc.png", caption="Confusion Matrix for SVC")
        st.image("images\svcroc.png", caption="ROC Curve for SVC")
    elif model_choice == "GNB":
        prediction = gnb.predict(input_data_scaled)
        st.image("images\gnb.png", caption="Confusion Matrix for GNB")
        st.image("images\gnbroc.png", caption="ROC Curve for GNB")
    elif model_choice == "Decision Tree":
        prediction = griddt.predict(input_data_scaled)
        st.image("images\dt.png", caption="Confusion Matrix for Decision Tree")
        st.image("images\dtroc.png", caption="ROC Curve for Decision Tree")
    elif model_choice == "Random Forest":
        prediction = gridrfc.predict(input_data_scaled)
        st.image("images\_rfc.png", caption="Confusion Matrix for Random Forest")
        st.image("images\_rfcroc.png", caption="ROC Curve for Random Forest")
    elif model_choice == "AdaBoost":
        prediction = ada.predict(input_data_scaled)
        st.image("images\_ada.png", caption="Confusion Matrix for AdaBoost")
        st.image("images\_adaroc.png", caption="ROC Curve for AdaBoost")
    elif model_choice == "Gradient Boosting":
        prediction = gbc.predict(input_data_scaled)
        st.image("images\gbc.png", caption="Confusion Matrix for Gradient Boosting")
        st.image("images\gbcroc.png", caption="ROC Curve for Gradient Boosting")
    elif model_choice == "XGBoost":
        prediction = xgb.predict(input_data_scaled)
        st.image("images\_xgb.png", caption="Confusion Matrix for XGBoost")
        st.image("images\_xgbroc.png", caption="ROC Curve for XGBoost")

    # Display the result
    if prediction == 1:
        st.write("Prediction: Income > 50K")
    else:
        st.write("Prediction: Income <= 50K")
