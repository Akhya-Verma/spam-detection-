import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app title and description
st.title("Spam Detection Web App")
st.write("This is a web app to classify messages as spam or ham (not spam).")

# Input text field
user_input = st.text_area("Enter the message you want to classify")

# Button to trigger prediction
if st.button("Predict"):
    if user_input:
        # Preprocess input and predict
        user_input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(user_input_tfidf)
        result = 'Spam' if prediction[0] == 1 else 'Ham'

        # Display the prediction result
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter a message to classify.")
