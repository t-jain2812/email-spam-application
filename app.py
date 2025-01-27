import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Email Spam Classification Application")
st.write("This application uses a machine learning model to classify emails as spam or not spam.")
user_input = st.text_area("Enter an email to classify", height= 150)

if st.button("classify"):
    if user_input:
        data=[user_input]
        vectorizer_data = cv.transform(data).toarray()
        result = model.predict(vectorizer_data)
        if result[0] == 0:
            st.write("The email is not spam")
        else:
            st.write("The email is spam")
    else:
        st.write("Please enter an email to classify")