import streamlit as st

st.title("Model Training")
mlTab, nnTab = st.tabs(["Machine Learning", "Neural Networks"])
with mlTab:
    st.header("Random Forest Classifier Training")
    st.subheader("Data Scailing and Dividing")
    st.code("""
    #devided data for training column and answer column
    rfcX = kd.drop(columns=['Personality'])
    rfcy = kd['Personality']

    #scaling data
    scaler = MinMaxScaler()
    rfcX_scaled = pd.DataFrame(scaler.fit_transform(rfcX), columns=rfcX.columns)
    rfcX_scaled
    """)
    st.subheader("Training, Evalutaing, and Saving Model")
    st.code("""
    #split data for training and testing
    rfcX_train, rfcX_test, rfcy_train, rfcy_test = train_test_split(rfcX_scaled, rfcy, test_size=0.2, random_state=42)

    #model training
    rfc_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    rfc_model.fit(rfcX_train, rfcy_train)

    #model tesing
    rfcy_pred = rfc_model.predict(rfcX_test)

    #model evaluating
    rfc_acc = accuracy_score(rfcy_test, rfcy_pred)

    print(f"Random Forest Classifier Accuracy: {rfc_acc:.2f}\\n")

    #save
    pickle.dump(rfc_model, open('rfc_model.pkl', 'wb'))
    pickle.dump(scaler, open('rfc_scaler.pkl', 'wb'))
    """)

    st.header("SVM Training")
    st.subheader("Data Scailing and Dividing")
    st.code("""
    #devided data for training column and answer column
    svmX = kd.drop(columns=['Personality'])
    svmy = kd['Personality']

    #scaling data
    scaler = MinMaxScaler()
    svmX_scaled = pd.DataFrame(scaler.fit_transform(svmX), columns=svmX.columns)
    svmX_scaled
    """)
    st.subheader("Training, Evalutaing, and Saving Model")
    st.code("""
    #split data for training and testing
    svmX_train, svmX_test, svmy_train, svmy_test = train_test_split(svmX_scaled, svmy, test_size=0.2, random_state=42)

    #model training
    svm_model = SVC(kernel="poly", random_state=42, class_weight="balanced")
    svm_model.fit(svmX_train, svmy_train)
            
    #model tesing
    svmy_pred = svm_model.predict(svmX_test)

    #model evaluating
    svm_acc = accuracy_score(svmy_test, svmy_pred)

    print(f"SVM Accuracy: {svm_acc:.2f}\\n")

    #save
    pickle.dump(rfc_model, open('svm_model.pkl', 'wb'))
    pickle.dump(scaler, open('svm_scaler.pkl', 'wb'))
    """)

with nnTab:
    st.header("FNN Training")
    st.subheader("Data Scailing and Dividing")
    st.code("""
    #devided data for training column and answer column
    X = kd.drop(columns=['EngagementLevel', 'PlayTimeHours', 'PlayerID'])
    y = kd['EngagementLevel']

    #scaling data
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled
    """)
    st.subheader("Training, Evalutaing, and Saving Model")
    st.code("""
    #split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    #model training
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="softmax")
    ])
    # Use a lower learning rate for better convergence
    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    #model tesing
    history = model.fit(
        X_train, y_train,
        epochs=20, 
        batch_size = 16,
        validation_split = 0.2

    )

    #model evaluating
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    from math import ceil
    print(f"Batch : {ceil(((len(X_train) - int((len(X_train) * 0.2)))) / 32)}")
    print(f"Test accuracy: {test_accuracy:.2f}")

    #save
    model.save('fnn_model.keras')
    pickle.dump(scaler, open('fnn_scaler.pkl', 'wb'))
    """)