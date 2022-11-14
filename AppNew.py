import streamlit as st
from predict_new import show_new_predict_page
from EXPLOREEXAMPLE import show_explore_page


page = st.sidebar.selectbox("Explore the Data Or Classify a Tumor", ("Classify a Tumor", "Explore DATA"))

if page == "Classify a Tumor":
    show_new_predict_page()
else:
    show_explore_page()
