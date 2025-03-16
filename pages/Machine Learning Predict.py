import streamlit as st
import pickle
import pandas as pd
import random

with open('./training/rfc_model.pkl', "rb") as rfc_file:
    rfc_model = pickle.load(rfc_file)
with open('./training/rfc_scaler.pkl', "rb") as rfc_scale:
    rfc_scaler = pickle.load(rfc_scale)
with open('./training/rfc_model.pkl', "rb") as svm_file:
    svm_model = pickle.load(svm_file)
with open('./training/rfc_scaler.pkl', "rb") as svm_scale:
    svm_scaler = pickle.load(svm_scale)

def map_data(x):
    convert = x.unique()
    return x.map(dict(zip(convert, range(1,len(convert) + 1))))

def get_mbti(number):
    if number == 1:
        return "ENTP"
    elif number == 2:
        return "INTP"
    elif number == 3:
        return "ESFP"
    elif number == 4:
        return "ENFJ"
    elif number == 5:
        return "ISFP"
    elif number == 6:
        return "ISFJ"
    elif number == 7:
        return "ESTJ"
    elif number == 8:
        return "INFP"
    elif number == 9:
        return "ENFP"
    elif number == 10:
        return "ESTP"
    elif number == 11:
        return "ESFJ"
    elif number == 12:
        return "ISTJ"
    elif number == 13:
        return "INTJ"
    elif number == 14:
        return "INFJ"
    elif number == 15:
        return "ISTP"
    elif number == 16:
        return "ENTJ"
    else:
        return "Invalid number"
        
st.title("✨ MBTI Prediction ✨")
if "age" not in st.session_state:
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
gender = st.radio("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.gender))
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
input_data = pd.DataFrame([[age, gender, education, introversionScore, sensingScore, thinkingScore, judgingScore, interest]], columns=["Age", "Gender", "Education", "Introversion Score", "Sensing Score", "Thinking Score", "Judging Score", "Interest"])
# Mapping data
for column in input_data.columns:
    if input_data[column].dtype == 'object':
        input_data[column] = map_data(input_data[column])
input_data = input_data.astype(int)

rfcTab, svmTab = st.tabs(["Random Forest Classifier", "Support Vector Machine"])
with rfcTab:
    input_scaled = rfc_scaler.fit_transform(input_data)

    # พยากรณ์ MBTIเมื่อกดปุ่ม
    if st.button("✨ Predict with RFC! ✨"):
        prediction = rfc_model.predict(input_data)
        st.success(f"🎭 Predicted MBTI: {get_mbti(prediction[0])} 🎉")
with svmTab:
    input_scaled = rfc_scaler.fit_transform(input_data)

    # พยากรณ์ MBTIเมื่อกดปุ่ม
    if st.button("✨ Predict with SVM! ✨"):
        prediction = rfc_model.predict(input_data)
        st.success(f"🎭 Predicted MBTI: {get_mbti(prediction[0])} 🎉")