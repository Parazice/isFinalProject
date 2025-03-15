import streamlit as st
import pickle
import pandas as pd

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
# รับค่าจากผู้ใช้
age = st.number_input("Age", min_value=12, max_value=120, step=1)
gender = st.radio("Gender", ["Male", "Female"])
educationPick = st.radio("Education", ["Yes", "No"])
introversionScore = st.number_input("Introvertsion Score", min_value=0.00, max_value=10.00, step=0.01)
sensingScore = st.number_input("Sensing Score", min_value=0.00, max_value=10.00, step=0.01)
thinkingScore = st.number_input("Thinking Score", min_value=0.00, max_value=10.00, step=0.01)
judgingScore = st.number_input("Judging Score", min_value=0.00, max_value=10.00, step=0.01)
interest = st.selectbox("Interest", ['Arts', 'Sports', 'Technology', 'Others', 'Unknown'])

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
for col in input_data.columns:
    print(input_data[col].unique())
input_data = input_data.astype(int)

rfcTab, svmTab = st.tabs(["Random Forest Classifier", "Support Vector Machine"])
with rfcTab:
    input_scaled = rfc_scaler.fit_transform(input_data)

    # พยากรณ์ MBTIเมื่อกดปุ่ม
    if st.button("✨ Predict with RFC! ✨"):
        prediction = rfc_model.predict(input_data)
        st.success(f"🎭 Predicted MBTI: {get_mbti(prediction[0])} 🎉")
        st.markdown(
            """
            <style>
            div.stAlert {
                font-size: 36px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
with svmTab:
    input_scaled = rfc_scaler.fit_transform(input_data)

    # พยากรณ์ MBTIเมื่อกดปุ่ม
    if st.button("✨ Predict with SVM! ✨"):
        prediction = rfc_model.predict(input_data)
        st.success(f"🎭 Predicted MBTI: {get_mbti(prediction[0])} 🎉")
        st.markdown(
            """
            <style>
            div.stAlert {
                font-size: 36px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )