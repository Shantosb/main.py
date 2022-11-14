import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('rs_randomforestModel.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

randomforest = data


def show_new_predict_page():
    st.markdown("<h1 style='text-align: center; color: White;'>Tumor Diagnosis</h1>", unsafe_allow_html=True)

    st.write("""### Input all the recuired information below:""")

    radius_mean = st.number_input("enter radius mean")
    st.markdown(f"Radius mean: {radius_mean}")

    texture_mean = st.number_input("enter texture mean")
    st.markdown(f"texture mean: {texture_mean}")

    perimeter_mean = st.number_input("enter perimeter mean")
    st.markdown(f"perimeter mean: {perimeter_mean}")

    area_mean = st.number_input("enter area_mean")
    st.markdown(f"area mean: {area_mean}")

    smoothness_mean = st.number_input("enter smoothness mean")
    st.markdown(f"smoothness mean: {smoothness_mean}")

    compactness_mean = st.number_input("enter compactness mean")
    st.markdown(f"compactness mean: {compactness_mean}")

    concavity_mean = st.number_input("enter concavity mean")
    st.markdown(f"concavity mean: {concavity_mean}")

    concave_points_mean = st.number_input("enter concave points mean")
    st.markdown(f"concave points mean: {concave_points_mean}")

    symmetry_mean = st.number_input("enter symmetry mean")
    st.markdown(f"symmetry mean: {symmetry_mean}")

    radius_se = st.number_input("enter radius se")
    st.markdown(f"radius se: {radius_se}")

    perimeter_se = st.number_input("enter perimeter se")
    st.markdown(f"perimeter se: {perimeter_se}")

    area_se = st.number_input("enter area se")
    st.markdown(f"area_se: {area_se}")

    compactness_se = st.number_input("enter compactness se")
    st.markdown(f"compactness se: {compactness_se}")

    concavity_se = st.number_input("enter concavity se")
    st.markdown(f"concavity se: {concavity_se}")

    concave_points_se = st.number_input("enter concave points se")
    st.markdown(f"concave points se: {concave_points_se}")

    radius_worst = st.number_input("enter radius worst")
    st.markdown(f"radius worst: {radius_worst}")

    texture_worst = st.number_input("enter texture worst")
    st.markdown(f"texture worst: {texture_worst}")

    perimeter_worst = st.number_input("enter perimeter worst")
    st.markdown(f"perimeter worst: {perimeter_worst}")

    area_worst = st.number_input("enter area worst")
    st.markdown(f"area worst: {area_worst}")

    smoothness_worst = st.number_input("enter smoothness worst")
    st.markdown(f"smoothness worst: {smoothness_worst}")

    compactness_worst = st.number_input("enter compactness worst")
    st.markdown(f"compactness worst: {compactness_worst}")

    concavity_worst = st.number_input("enter concavity worst")
    st.markdown(f"concavity worst: {concavity_worst}")

    concave_points_worst = st.number_input("enter concave points worst")
    st.markdown(f"concave points worst: {concave_points_worst}")

    symmetry_worst = st.number_input("enter symmetry_worst")
    st.markdown(f"symmetryworst: {symmetry_worst}")

    fractal_dimension_worst = st.number_input("enter fractal_dimension_worst")
    st.markdown(f"fractal dimension worst: {fractal_dimension_worst}")

    list = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
            symmetry_mean, radius_se, perimeter_se, area_se, compactness_se, concavity_se, concave_points_se, radius_worst,
            texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
            symmetry_worst,fractal_dimension_worst]

    x = np.array(list)
    x = x.reshape(1, -1)

    st.write("#")
    st.write("#")
    col1, col2 = st.columns(2)
    with col1:
        ok = st.button("Submit")
    if ok:

        diagnosis = randomforest.predict(x)
        if diagnosis == 1:
            st.markdown(f"Tumor is Malignant")
        else:
            st.markdown(f"Tumor is Benign")
    with col2:
        probability_button = st.button("Check Probability")

    diagnosis_proba = randomforest.predict_proba(x)

    if probability_button:
        ix = diagnosis_proba.argmax(1).item()
        list = []
        list.append(f'{diagnosis_proba[0, ix]:.2%}')
        st.markdown(f"The probability of a correct diagnosis on this sample is: {list}")