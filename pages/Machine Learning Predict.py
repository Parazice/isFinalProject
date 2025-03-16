import streamlit as st
import pickle
import pandas as pd
import random
from tensorflow.keras.models import load_model

@st.cache_resource
def load_rfc_model():
    with open("./training/rfc_model.pkl", "rb") as f:
        return pickle.load(f)

rfc_model = load_rfc_model()

@st.cache_resource
def load_rfc_scaler():
    with open("./training/svm_scaler.pkl", "rb") as f:
        return pickle.load(f)
    
rfc_scaler = load_rfc_scaler()   

@st.cache_resource
def load_svm_model():
    with open("./training/svm_model.pkl", "rb") as f:
        return pickle.load(f)

svm_model = load_svm_model()

@st.cache_resource
def load_svm_scaler():
    with open("./training/svm_scaler.pkl", "rb") as f:
        return pickle.load(f)
    
svm_scaler = load_svm_scaler()

def map_data(column, value):
    mapping_dict = {
        "Gender": {"Male": 0, "Female": 1},
        "Interest": {'Arts': 0, 'Technology': 1, 'Sports': 2, 'Others': 3, 'Unknown': 4},
        "Personality": {'ISFP': 0, 'ISFJ': 1, 'INFP': 2, 'ESFJ': 3, 'INTJ': 4, 'INFJ': 5, 'ESTP': 6, 'ISTJ': 7,
       'ISTP': 8, 'ENTP': 9, 'ESFP': 10, 'INTP': 11, 'ENTJ': 12, 'ESTJ': 13, 'ENFJ': 14, 'ENFP': 15}
    }
    return mapping_dict[column].get(value)
def get_MBTI(number):
    mbti = ['ISFP', 'ISFJ', 'INFP', 'ESFJ', 'INTJ', 'INFJ', 'ESTP', 'ISTJ',
       'ISTP', 'ENTP', 'ESFP', 'INTP', 'ENTJ', 'ESTJ', 'ENFJ', 'ENFP']
    return mbti[number]
    
        
st.title("✨ MBTI Prediction ✨")
if "educationPick" not in st.session_state:
    st.session_state.age = 12
    st.session_state.gender = "Male"
    st.session_state.educationPick = "Yes"
    st.session_state.introversionScore = 0.00
    st.session_state.sensingScore = 0.00
    st.session_state.thinkingScore = 0.00
    st.session_state.judgingScore = 0.00
    st.session_state.interest = "Unknown"

# ฟังก์ชันสุ่มค่า
def randomize_values():
    st.session_state.age = random.randint(12, 60)
    st.session_state.gender = random.choice(["Male", "Female"])
    st.session_state.educationPick = random.choice(["Yes", "No"])
    st.session_state.introversionScore = random.uniform(0, 10)
    st.session_state.sensingScore = random.uniform(0, 10)
    st.session_state.thinkingScore = random.uniform(0, 10)
    st.session_state.judgingScore = random.uniform(0, 10)
    st.session_state.interest = random.choice(['Arts', 'Sports', 'Technology', 'Others', 'Unknown'])
st.button("🎲 Randomize", on_click=randomize_values)

# รับค่าจากผู้ใช้
age = st.number_input("Age", min_value=12, max_value=60, step=1, value=int(st.session_state.age))
gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.gender))
educationPick = st.radio("Education", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.educationPick))
introversionScore = st.number_input("Introversion Score", min_value=0.00, max_value=10.00, step=0.01, value=float(st.session_state.introversionScore))
sensingScore = st.number_input("Sensing Score", min_value=0.00, max_value=10.00, step=0.01, value=float(st.session_state.sensingScore))
thinkingScore = st.number_input("Thinking Score", min_value=0.00, max_value=10.00, step=0.01, value=float(st.session_state.thinkingScore))
judgingScore = st.number_input("Judging Score", min_value=0.00, max_value=10.00, step=0.01, value=float(st.session_state.judgingScore))
interest = st.selectbox("Interest", ['Arts', 'Sports', 'Technology', 'Others', 'Unknown'], index=['Arts', 'Sports', 'Technology', 'Others', 'Unknown'].index(st.session_state.interest))

if educationPick == "Yes":
    education = 1
else:
    education = 0

# ทำเป็น DataFrame เพื่อส่งเข้าโมเดล
input_data = pd.DataFrame([[age, gender, education, introversionScore, sensingScore, thinkingScore, judgingScore, interest]], 
                          columns=["Age", "Gender", "Education", "Introversion Score", "Sensing Score", "Thinking Score", "Judging Score", "Interest"])
# Mapping data
for column in input_data.columns:
    if input_data[column].dtype == 'object':
        print(column)
        input_data[column] = input_data[column].apply(lambda x: map_data(column, x))
kd = input_data.astype(int)

rfcTab, svmTab = st.tabs(["Random Forest Classifier", "Support Vector Machine"])
with rfcTab:
    input_scaled = rfc_scaler.transform(input_data)

    # พยากรณ์ MBTIเมื่อกดปุ่ม
    if st.button("✨ Predict with RFC! ✨"):
        prediction = rfc_model.predict(input_scaled)
        st.success(f"🎭 Predicted MBTI: {get_MBTI(prediction[0])} 🎉")
with svmTab:
    input_scaled = svm_scaler.transform(input_data)

    # พยากรณ์ MBTIเมื่อกดปุ่ม
    if st.button("✨ Predict with SVM! ✨"):
        prediction = rfc_model.predict(input_scaled)
        st.success(f"🎭 Predicted MBTI: {get_MBTI(prediction[0])} 🎉")