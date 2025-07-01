import streamlit as st
import joblib

model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('text_vectorizer.pkl')

st.title("SMS Spam Classification")
st.write("Enter a message and let the model predict if its **SPAM** or **HAM**")

user_input = st.text_area("Type your message here: ")

if st.button("classify"):
    if user_input.strip() == "":
        st.warning("Please enter a Message")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction =model.predict(vectorized_input)[0]
        prediction_prob = model.predict_proba(vectorized_input).max()

        if prediction == "Spam":
            st.error(f"Prediction: **Spam** ({prediction_prob: .2%} confidence)")
        else:
            st.success(f"Prediction: **Spam** ({prediction_prob: .2%} confidence)")
