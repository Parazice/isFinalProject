import streamlit as st
pages = {
    "Developing": [
        st.Page("./pages/Dataset Prepare.py", title="Dataset Preparing"),
        st.Page("./pages/Model Training.py", title="Model Training"),
    ],
    "Demo": [
        st.Page("./pages/Machine Learning Predict.py", title="Machine Learning"),
        st.Page("./pages/Neural Networks Predict.py", title="Neural Network"),
    ]
    # "Reference": [
    #     st.Page("views/reference_page.py", title="Reference"),
    # ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()