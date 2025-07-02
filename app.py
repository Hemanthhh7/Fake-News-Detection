import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
fake_df['label'] = 0
true_df['label'] = 1
df = pd.concat([fake_df, true_df]).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text_tokens = text.split()
    filtered_words = [w for w in text_tokens if not w in stop_words]
    return ' '.join(filtered_words)

df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

st.title("📰 Fake News Classifier")
st.write("Paste a news article below to classify it as FAKE or REAL.")

user_input = st.text_area("Your News Text:", height=300)

if st.button("Predict"):
    cleaned_input = clean_text(user_input)
    input_tfidf = tfidf.transform([cleaned_input])
    prediction = model.predict(input_tfidf)[0]
    if prediction == 0:
        st.error("⚠️ This news is predicted to be FAKE.")
    else:
        st.success("✅ This news is predicted to be REAL.")
