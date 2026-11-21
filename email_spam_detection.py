from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import spam_utils  # noqa: F401  (needed so joblib can unpickle the pipeline)
from spam_utils import FEATURES


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).with_name("spam_model.joblib"))


def main():
    html_temp = """
        <h1>Email Spam Detection</h1>
    """

    model = load_model()

    st.markdown(html_temp, unsafe_allow_html=True)

    st.markdown("This app will help you to detect whether an email is spam")

    p1 = st.number_input("Please enter email length (In Characters)", 0, 1000, 100, step=1)

    p2 = st.number_input("Please enter number of links", 0, 100, 1, step=1)

    p3 = st.number_input("Please enter number of special characters", 0, 100, 3, step=1)

    p4 = st.number_input("Please enter number of capital words", 0, 100, 5, step=1)

    s1 = st.selectbox("Does the email have an attachment?", ("No", "Yes"))

    if s1 == 'No':
        p5 = 0
    elif s1 == 'Yes':
        p5 = 1

    data_new = pd.DataFrame({
        'Email_Length': p1,
        'Num_Links': p2,
        'Num_Special_Chars': p3,
        'Capital_Words': p4,
        'Has_Attachment': p5,
    }, index=[0])[FEATURES]

    if st.button("Predict"):
        pred = model.predict(data_new)[0]
        prob = model.predict_proba(data_new)[0][1]
        if pred == 1:
            st.error("This email looks like SPAM (spam probability: {:.1f}%)".format(prob * 100))
        else:
            st.success("This email looks safe, not spam (spam probability: {:.1f}%)".format(prob * 100))


if __name__ == '__main__':
    main()
