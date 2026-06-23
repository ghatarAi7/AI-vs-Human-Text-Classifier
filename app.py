import streamlit as st
import pickle

# تحميل الموديل
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("🤖 AI vs Human Text Detector")

st.write("اكتب أي نص والنظام يحدد هل هو من إنسان أو ذكاء اصطناعي")

text = st.text_area("✍️ اكتب النص هنا:")

if st.button("تحليل"):
    if text.strip() == "":
        st.warning("اكتب نص أول")
    else:
        vector = vectorizer.transform([text])
        pred = model.predict(vector)
        proba = model.predict_proba(vector)

        confidence = max(proba[0]) * 100

        if pred[0] == 1:
            st.error("🤖 النص مولد بالذكاء الاصطناعي")
        else:
            st.success("👤 النص بشري")

        st.info(f"📊 نسبة الثقة: {confidence:.2f}%")