# Main page interface of the breast cancer prediction application
# Import necessary libraries/modules/packages which may be of need for the app implementation

# The application will be implemented wholly using third-party, open source Streamlit package in Python
# Import streamlit and all its components.

# Since working on a virtual environment requires some packages to be installed,
# the installation of these packages has been performed individually, inside the environment's folder.

# Otherwise, globally run them (since all have been installed globally/built in)

import streamlit as st

# Import numpy Python package is useful for mathematical operations or experiments with the scikit-learn module.
# (used later on)

import numpy as np

# Import pandas Python package, for Data Science operations/tasks
# as well as CSV datasets uploaded and read by the user.

import pandas as pd

# Import sklearn module (or scikit-learn) since most experiments will base on machine learning/data science

# The scikit-learn module will be called and imported using the "sklearn" abbreviation
# and by importing it, all features (except sub-components on processes) will be granted access.

import sklearn

# Import built-in math modules

# Import csv module used to recognize .csv files read by the user.

import csv

# Import built-in math module
import math

# Import built-in random package for randomness in numbers and/or time.
import random

# Import built-in scipy package will be imported with all its features.
import scipy

# Import plotting packages such as matplotlib and seaborn

# Import matplotlib module for basic plotting and histogram creation
import matplotlib.pyplot as plt

# Import seaborn module for more advanced plotting and heatmap usage
import seaborn as sns

# Import necessary submodules

# Import built-in joblib module for tasks using pipelining
import joblib

# Import built-in pickle module for loading pre-trained models
import pickle

# Import built-in os module in case the pre-trained model is saved to a directory
# (and to avoid errors)

import os

# import warnings
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Main code

# Load pre-trained model

# specify paths (optional)
# model_load_path = 'C:\\Users\\user\\Documents\\thesis\\dtc_down.pkl'

# load the model
dtc_loaded = joblib.load('dtc_down.pkl')

# Define class weights based on knowledge of class distribution
class_weights = {
    1: 1,  # class weights for negative samples
    2: 1   # class weights for positive samples
}

# using a path (optional)
# dtc_loaded = joblib.load(model_load_path)

# Now you can use the loaded model for prediction or other tasks

# Add a title to the app
st.title("Breast Cancer Prediction App")

# Add a welcome text
st.write("## Welcome to BreastDiagApp!")

# Add a message that asks user for input as so
st.write("### Please enter values to proceed.")

# Add a description
st.write("No registration required!")

# examine columns one by one

# Add input values, for the user to insert desired values into the app
# for the prediction to be made.

# Add a message that asks user for input as so
st.write("**Enter values in the options below to predict the result.**")

# Create a function, which aims to get input for a feature

def get_feature_input(feature_name):
    return st.number_input(feature_name)

# name features for user input where necessary

features = [
    'Age', 'BMI', 'Glucose', 'HOMA', 'Insulin', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1'
]

# Initialize an empty dictionary to store feature values
feature_names = {}

# Iterate over each feature and create input fields
for feature in features:
    if feature == 'Age':
        feature_names[feature] = st.number_input(feature, step=1, format="%d")
    else:

        # get feature names by function

        # feature_names[feature] = st.number_input(feature)
        feature_names[feature] = get_feature_input(feature)

# Once all features are input, display the feature vector
# if st.button("Create Feature Vector"):
#    feature_vector = [feature_names[feature] for feature in features]
#    st.write("Feature vector:", feature_vector)

# Button to make prediction
if st.button("Predict"):

    # Convert feature values to array
    feature_vector = np.array([list(feature_names.values())])
    
    # Make prediction probabilities
    prediction_prob = dtc_loaded.predict_proba(feature_vector)

    # Make prediction
    # prediction = dtc_loaded.predict(feature_vector)

    # Adjust decision threshold
    threshold = 0.5  # Adjust this threshold as needed
    predicted_class = 1 if prediction_prob[0, 1] > threshold else 0

    # Display prediction result
    if predicted_class == 1:
        st.write("Prediction: Negative")
    else:
        st.write("Prediction: Positive")

# Slider to adjust threshold
threshold = st.slider("Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# stuff will be added