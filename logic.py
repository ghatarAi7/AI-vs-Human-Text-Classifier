import pickle
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

DATA_PATH = "AI_Human.csv"

df = pd.read_csv(DATA_PATH)

print("Columns:", df.columns)

text_column = "text"
label_column = "generated"

df = df[[text_column, label_column]].dropna()

df = df.sample(n=min(20000, len(df)), random_state=42)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df[text_column] = df[text_column].apply(clean_text)

X = df[text_column]
y = df[label_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

print("\nResults:")
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel saved successfully.")