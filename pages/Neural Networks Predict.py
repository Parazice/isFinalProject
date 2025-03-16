import streamlit as st
import pickle
import pandas as pd
import random
import keras
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model_from_file():
    return load_model("./training/fnn_model.keras")

fnn_model = load_model_from_file()
@st.cache_resource
def load_scaler_from_file():
    with open("./training/fnn_scaler.pkl", "rb") as f:
        return pickle.load(f)
def map_data(column, value):
    mapping_dict = {
        "Gender": {"Male": 0, "Female": 1},
        "Location": {"USA": 0, "Europe": 1, "Asia": 2, "Other": 3},
        "GameGenre": {'Strategy': 0, 'Simulation': 1, 'Action': 2, 'RPG': 3, 'Sports': 4},
        "GameDifficulty": {'Easy': 0, 'Medium': 1, 'Hard': 2}
    }
    return mapping_dict[column].get(value, -1)
def get_engLevel(prediction):
    highest_index = prediction[0].argmax()
    engage_levels = ["Low", "Normal", "High"]
    predicted_level = engage_levels[highest_index]
    return predicted_level

fnn_scaler = load_scaler_from_file()

st.title("🎮Gaming Engagement Predict🎮")

if "location" not in st.session_state:
    st.session_state.age = 12
    st.session_state.gender = "Male"
    st.session_state.location = "USA"
    st.session_state.genre = "Strategy"
    st.session_state.purchase = "Yes"
    st.session_state.difficulty = "Easy"
    st.session_state.session = 0
    st.session_state.duration = 0
    st.session_state.level = 0
    st.session_state.achieve = 0

# ฟังก์ชันสุ่มค่า
def randomize_values():
    st.session_state.age = random.randint(12, 60)
    st.session_state.gender = random.choice(["Male", "Female"])
    st.session_state.location = random.choice(["USA", "Europe", "Asia", "Other"])
    st.session_state.genre = random.choice(['Strategy', 'Simulation', 'Action', 'RPG', 'Sports'])
    st.session_state.purchase = random.choice(["Yes", "No"])
    st.session_state.difficulty = random.choice(['Easy', 'Medium', 'Hard'])
    st.session_state.session = random.randint(0,20)
    st.session_state.duration = random.randint(0,240)
    st.session_state.level = random.randint(0,99)
    st.session_state.achieve = random.randint(0,50)
st.button("🎲 Randomize", on_click=randomize_values)

age = st.number_input("Age", min_value=12, max_value=60, step=1, value=int(st.session_state.age))
gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.gender))
location = st.selectbox("Location", ["USA", "Europe", "Asia", "Other"], index = ["USA", "Europe", "Asia", "Other"].index(st.session_state.location))
genre = st.selectbox("Game Genre", ['Strategy', 'Simulation', 'Action', 'RPG', 'Sports'], index = ['Strategy', 'Simulation', 'Action', 'RPG', 'Sports'].index(st.session_state.genre))
purchase = st.radio("In-game Purchases", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.purchase))
difficulty = st.selectbox("Difficulty", ['Easy', 'Medium', 'Hard'], index = ['Easy', 'Medium', 'Hard'].index(st.session_state.difficulty))
session = st.number_input("Sessions per week", min_value=0, max_value=20, step=1, value=int(st.session_state.session))
duration = st.number_input("Average Session Duration (Minutes)", min_value=0, max_value=240, value=int(st.session_state.duration))
level = st.number_input("Player Level", min_value=0, max_value=99, value=int(st.session_state.level))
achieve = st.number_input("Achievements", min_value=0, max_value=50, value=int(st.session_state.achieve))

if purchase == "Yes":
    purchase = 1
else:
    purchase = 0

input_data = pd.DataFrame([[age, gender, location, genre, purchase, difficulty, session, duration, level, achieve]], 
                          columns=['Age', 'Gender', 'Location', 'GameGenre', 'InGamePurchases', 'GameDifficulty', 
                                   'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked'])

for column in input_data.columns:
    if input_data[column].dtype == 'object':
        input_data[column] = input_data[column].apply(lambda x: map_data(column, x))
input_data = input_data.astype(int)


if st.button("✨ Predict with FNN! ✨"):
        input_scaled = fnn_scaler.transform(input_data)
        prediction = fnn_model.predict(input_scaled)
        st.success(f"🎮 Predicted Gaming Engagement Level: {get_engLevel(prediction)} 🎉")