import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st  # Import Streamlit

# Load and prepare data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

# Title and description for the UI
st.title("Student Marks Prediction App")
st.write("This app predicts the marks based on the number of hours a student studies using Linear Regression.")

# Show data preview
if st.checkbox("Show raw data"):
    st.write(data.head())

# Train the model
x = data[['Hours']]
y = data['Scores']
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

regressor = LinearRegression()
regressor.fit(train_x, train_y)

# Input section in the UI for custom hours input
st.write("## Enter the number of hours the student studies:")
hours = st.number_input("Hours studied per day", min_value=0.0, max_value=24.0, value=1.0)

# Prediction button
if st.button("Predict Score"):
    prediction = regressor.predict([[hours]])
    st.write(f"A student who studies for {hours} hours is estimated to score {prediction[0]:.2f}")

# Visualization
st.write("## Scatterplot of Hours vs Scores")
sns.scatterplot(x=data['Hours'], y=data['Scores'])
sns.regplot(x=data['Hours'], y=data['Scores'])
st.pyplot(plt)

# Display Actual vs Predicted
st.write("## Actual vs Predicted on Validation Set")
pred_y = regressor.predict(val_x)
comparison_df = pd.DataFrame({'Actual': val_y, 'Predicted': pred_y})
st.write(comparison_df)

# Accuracy of the model
st.write('### Model Accuracy')
st.write(f"Train accuracy: {regressor.score(train_x, train_y):.2f}")
st.write(f"Test accuracy: {regressor.score(val_x, val_y):.2f}")
