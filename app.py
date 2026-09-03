import streamlit as st
import joblib
import numpy as np

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction")

preg=st.number_input("Pregnancies")
glu=st.number_input("Glucose")
bp=st.number_input("Blood Pressure")
skin=st.number_input("Skin Thickness")
ins=st.number_input("Insulin")
bmi=st.number_input("BMI")
dpf=st.number_input("Diabetes Pedigree Function")
age=st.number_input("Age")

if st.button("Predict"):

    data=np.array([[preg,glu,bp,skin,ins,bmi,dpf,age]])

    data=scaler.transform(data)

    prediction=model.predict(data)

    probability=model.predict_proba(data)[0][1]

    if prediction[0]==1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")

    st.write("Probability:",round(probability*100,2),"%")