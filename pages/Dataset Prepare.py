import streamlit as st
import pandas as pd

st.title("Dataset Preparation")
mlTab, nnTab = st.tabs(["Machine Learning", "Neural Networks"])
with mlTab:
    st.header("Find the dataset")
    st.write("หา Dataset ที่สนใจจะนำมาทำ Machine Learning ในการคาดการณ์ข้อมูล โดยผมหามาจาก [kaggle.com](https://www.kaggle.com/)\n")
    st.write("[Click to download my dataset](https://www.kaggle.com/datasets/stealthtechnologies/predict-people-personality-types)")

    df = pd.DataFrame(pd.read_csv("./training/data.csv"))
    st.subheader("Predict People Personality Types Dataset")
    st.dataframe(df)

    st.subheader("Comlumn Description")
    columns = df.columns
    description = pd.DataFrame(
        {
            "Column Name": list(columns),
            "Description": [
                "A continuous variable representing the age of the individual.",
                "A categorical variable indicating the gender of the individual. Possible values are 'Male' and 'Female'.",
                "A binary variable, A value of 1 indicates the individual has at least a graduate-level education (or higher), and 0 indicates an undergraduate, high school level or Uneducated.",
                "A categorical variable representing the individual's primary area of interest.",
                "A continuous variable ranging from 0 to 10, representing the individual's tendency toward introversion versus extraversion. Higher scores indicate a greater tendency toward extraversion.",
                "A continuous variable ranging from 0 to 10, representing the individual's preference for sensing versus intuition. Higher scores indicate a preference for sensing.",
                "A continuous variable ranging from 0 to 10, indicating the individual's preference for thinking versus feeling. Higher scores indicate a preference for thinking.",
                "A continuous variable ranging from 0 to 10, representing the individual's preference for judging versus perceiving. Higher scores indicate a preference for judging.",
                "Target that contains People Personality Type"
            ],
            "Example": [
                df[i].unique() if df[i].dtypes == "object" else df[i].iloc[0]
                for i in df.columns
            ],
        }
    )
    st.dataframe(description, hide_index=True)

    st.header("Making Dataset Incomplete")
    st.write("เนื่องจาก Dataset ที่ดาวน์โหลดมานั้นเป็น Dataset ที่สมบูรณ์ ผมจึงใช้โค้ดนี้ในการทำให้ Dataset เกิดความเสียหายด้วยการทำให้ข้อมูลบางส่วนเป็น Null")
    st.code("""
    import pandas as pd
    import numpy as np

    # Load dataset
    file_path = "data.csv"
    df = pd.read_csv(file_path)

    # Set a percentage of missing values (e.g., 10% of the data)
    missing_percentage = 0.15
    num_missing = int(missing_percentage * df.size)

    # Randomly select positions to set as NaN
    np.random.seed(42)  # For reproducibility
    missing_indices = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    missing_indices = np.random.choice(len(missing_indices), num_missing, replace=False)

    for idx in missing_indices:
        row, col = divmod(idx, df.shape[1])
        df.iat[row, col] = np.nan

    # Save the modified dataset
    modified_file_path = "data_missing.csv"
    df.to_csv(modified_file_path, index=False)

    # Return the path of the modified file
    modified_file_path

    """)
    st.subheader("Dataset After Add Null Values")
    df1 = pd.DataFrame(pd.read_csv("./training/data_missing.csv"))
    st.dataframe(df1)

    st.header("Data Cleaning")
    st.subheader("Drop Null Values")
    st.code("""
    import pandas as pd
    import pickle
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
            
    kd = pd.read_csv('data_missing.csv')
    #checking Null values
    kd.isnull().any()

    #drop Null values
    kd = kd.dropna()
   
    """)
    st.subheader("Data Mapping")
    st.code("""
    def map_data(column, value):
        #dictionary for data mapping
        mapping_dict = {
            "Gender": {"Male": 0, "Female": 1},
            "Interest": {'Arts': 0, 'Technology': 1, 'Sports': 2, 'Others': 3, 'Unknown': 4},
            "Personality": {'ISFP': 0, 'ISFJ': 1, 'INFP': 2, 'ESFJ': 3, 'INTJ': 4, 'INFJ': 5, 'ESTP': 6, 'ISTJ': 7,
        'ISTP': 8, 'ENTP': 9, 'ESFP': 10, 'INTP': 11, 'ENTJ': 12, 'ESTJ': 13, 'ENFJ': 14, 'ENFP': 15}
        }
        return mapping_dict[column].get(value)
    for column in kd.columns:
        #if the value is string
        if kd[column].dtype == 'object':
            #replace with mapping dictionary's index
            kd[column] = kd[column].apply(lambda x: map_data(column, x))
    kd = kd.astype(int)
    kd
    """)
    st.write("เท่านี้ Dataset ก็พร้อมนำไปใช้ในการทำ Machine Learning แล้ว")

with nnTab:
    st.header("Find the dataset")
    st.write("หา Dataset ที่สนใจจะนำมาทำ Machine Learning ในการคาดการณ์ข้อมูล โดยผมหามาจาก [kaggle.com](https://www.kaggle.com/)\n")
    st.write("[Click to download my dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset/data)")

    df = pd.DataFrame(pd.read_csv("./training/online_gaming_behavior_dataset.csv"))
    st.subheader("🎮 Predict Online Gaming Behavior Dataset")
    st.dataframe(df)

    st.subheader("Comlumn Description")
    columns = df.columns
    description = pd.DataFrame(
        {
            "Column Name": list(columns),
            "Description": [
                "Unique identifier for each player.",
                "Age of the player.",
                "Gender of the player.",
                "Geographic location of the player.",
                "Genre of the game the player is engaged in.",
                "Average hours spent playing per session.",
                "Indicates whether the player makes in-game purchases (0 = No, 1 = Yes).",
                "Difficulty level of the game.",
                "Number of gaming sessions per week.",
                "Average duration of each gaming session in minutes.",
                "Current level of the player in the game.",
                "Number of achievements unlocked by the player.",
                "Categorized engagement level reflecting player retention ('High', 'Medium', 'Low')."
            ],
            "Example": [
                df[i].unique() if df[i].dtypes == "object" else df[i].iloc[0]
                for i in df.columns
            ],
        }
    )
    st.dataframe(description, hide_index=True)

    st.header("Making Dataset Incomplete")
    st.write("เนื่องจาก Dataset ที่ดาวน์โหลดมานั้นเป็น Dataset ที่สมบูรณ์ ผมจึงใช้โค้ดนี้ในการทำให้ Dataset เกิดความเสียหายด้วยการทำให้ข้อมูลบางส่วนเป็น Null")
    st.code("""
    import pandas as pd
    import numpy as np

    # Load dataset
    file_path = "./training/online_gaming_behavior_dataset.csv"
    df = pd.read_csv(file_path)

    # Set a percentage of missing values (e.g., 10% of the data)
    missing_percentage = 0.15
    num_missing = int(missing_percentage * df.size)

    # Randomly select positions to set as NaN
    np.random.seed(42)  # For reproducibility
    missing_indices = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    missing_indices = np.random.choice(len(missing_indices), num_missing, replace=False)

    for idx in missing_indices:
        row, col = divmod(idx, df.shape[1])
        df.iat[row, col] = np.nan

    # Save the modified dataset
    modified_file_path = "online_gaming_missing.csv"
    df.to_csv(modified_file_path, index=False)

    # Return the path of the modified file
    modified_file_path

    """)
    st.subheader("Dataset After Add Null Values")
    df1 = pd.DataFrame(pd.read_csv("./training/online_gaming_missing.csv"))
    st.dataframe(df1)

    st.header("Data Cleaning")
    st.subheader("Drop Null Values")
    st.code("""
    import pandas as pd
    import pickle
    import numpy as np
    import keras
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from keras import layers
    from keras.optimizers import Adam
    from keras.utils import to_categorical
            
    kd = pd.read_csv('online_gaming_missing.csv')
    #checking Null values
    kd.isnull().any()

    #drop Null values
    kd = kd.dropna()
   
    """)
    st.subheader("Data Mapping")
    st.code("""
    def map_data(column, value):
        mapping_dict = {
            "Gender": {"Male": 0, "Female": 1},
            "Location": {"USA": 0, "Europe": 1, "Asia": 2, "Other": 3},
            "GameGenre": {'Strategy': 0, 'Simulation': 1, 'Action': 2, 'RPG': 3, 'Sports': 4},
            "GameDifficulty": {'Easy': 0, 'Medium': 1, 'Hard': 2},
            "EngagementLevel": {'Low': 0, 'Medium': 1, 'High': 2}
        }
        return mapping_dict[column].get(value)
    for column in kd.columns:
        #if the value is string
        if kd[column].dtype == 'object':
            #replace with mapping dictionary's index
            kd[column] = kd[column].apply(lambda x: map_data(column, x))
    kd = kd.astype(int)
    kd
    """)
    st.write("เท่านี้ Dataset ก็พร้อมนำไปใช้ในการทำ Machine Learning แล้ว")
    
