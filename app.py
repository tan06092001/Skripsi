import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Load the model and label encoders
knn = joblib.load('knn_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load dataset
file_path = 'hero_data.csv'
df = pd.read_csv(file_path)

# Streamlit app title
st.title("Hero Recommendation System")

# User input fields
hero_role_input = st.selectbox("Select Hero Role", label_encoders['hero_role'].classes_)
hero_specially_input = st.selectbox("Select Hero Specialty", label_encoders['hero_specially'].classes_)
hero_durability_input = st.number_input("Hero Durability (0 - 100)", min_value=0)
hero_offence_input = st.number_input("Hero Offence (0 - 100)", min_value=0)
hero_ability_input = st.number_input("Hero Ability (0 - 100)", min_value=0)
hero_difficulty_input = st.number_input("Hero Difficulty (0 - 100)", min_value=0)

# Button to make predictions
if st.button("Recommend Heroes"):
    # Transform user input
    hero_role_encoded = label_encoders['hero_role'].transform([hero_role_input])[0]
    hero_specially_encoded = label_encoders['hero_specially'].transform([hero_specially_input])[0]

    # Prepare new data for prediction
    new_data = [[hero_role_encoded, hero_specially_encoded, hero_durability_input,
                 hero_offence_input, hero_ability_input, hero_difficulty_input]]

    # Find nearest neighbors
    distances, indices = knn.kneighbors(new_data)

    # Display results
    st.write("Recommended Heroes:")
    for i, index in enumerate(indices[0]):
        try:
            hero_name = df['hero_name'].iloc[index]  # Directly access the hero name
            st.write(f"{i + 1}. Hero: {hero_name}")
        except IndexError:
            st.write(f"{i + 1}. Hero: Unknown")

# Run the app
if __name__ == "__main__":
    pass  # Remove the st.run() line