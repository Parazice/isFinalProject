import streamlit as st

st.title("Model Training")
mlTab, nnTab = st.tabs(["Machine Learning", "Neural Networks"])
with mlTab:
    st.header("Get to know the algorithms")
    st.subheader("Random Forest Classifier")
    st.write(
        "A Random Forest is a collection of decision trees that work together to make predictions. In this article, we'll explain how the Random Forest algorithm works and how to use it."
    "")
    st.write("Random Forest algorithm is a powerful tree learning technique in Machine Learning to make predictions and then we do voting of all the tress to make prediction. They are widely used for classification and regression task.")
    st.write("How it works: FirstProcess starts with a dataset with rows and their corresponding class labels (columns).")
    st.write("Then - Multiple Decision Trees are created from the training data. Each tree is trained on a random subset of the data (with replacement) and a random subset of features. This process is known as bagging or bootstrap aggregating.")
    st.write("Each Decision Tree in the ensemble learns to make predictions independently.")
    st.write("When presented with a new, unseen instance, each Decision Tree in the ensemble makes a prediction.")
    st.divider()

    st.subheader("Support Ventor Machine (SVM)")
    st.write("Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. While it can handle regression problems, "
    "SVM is particularly well-suited for classification tasks."
    "SVM aims to find the optimal hyperplane in an N-dimensional space to separate data points into different classes. "
    "The algorithm maximizes the margin between the closest points of different classes.")
    st.write(
    "The key idea behind the SVM algorithm is to find the hyperplane that best separates two classes by maximizing the margin between them. "
    "This margin is the distance from the hyperplane to the nearest data points (support vectors) on each side.")
    st.divider()

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
    st.header("Get to know the algorithm")
    st.subheader("Feedforward Neural Networks (FNN)")
    st.write(
        "A Feedforward Neural Network (FNN) is a type of artificial neural network where connections between the nodes do not form cycles. This characteristic differentiates it from recurrent neural networks (RNNs). The network consists of an input layer, one or more hidden layers, and an output layer. Information flows in one direction—from input to output—hence the name \"feedforward."\
    "")
    st.write("Structure of a Feedforward Neural Network")
    st.write("1.Input Layer: The input layer consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.")
    st.write("2.Hidden Layers: One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.")
    st.write("3.Output Layer: The output layer provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.")
    st.divider()

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