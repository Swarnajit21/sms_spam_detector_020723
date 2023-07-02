import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

ps = PorterStemmer()

def transform_text(text):

  for x in text:
    x.lower()
  text = nltk.word_tokenize(text)

  y=[]

  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    k = ps.stem(i)
    y.append(k)

  return " ".join(y)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the messege")


if st.button('Predict'):
    # 1. preprocess
    transformed_text = transform_text(input_sms)
    # 2. vectorize
    vectorized_text = tfidf.transform([transformed_text])
    # 3. predict
    result = model.predict(vectorized_text)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
