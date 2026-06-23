# AI vs Human Text Classifier

A machine learning project that classifies text as human-written or AI-generated.

## Team Members

* Ghatar Butti Alqahtani
* Abdullah Saad Alqarni
* Albaraa Aldosari

## Tools Used

* Python
* Pandas
* Scikit-learn
* NLTK
* TF-IDF
* Logistic Regression
* Streamlit

## Files

* `logic.py` : Model training and prediction logic
* `app.py` : Streamlit interface
* `AI_Human.csv` : Dataset used locally for training. The full dataset is not included in this repository because of its size.

## Observation

The model can be affected by writing style. Highly formal and structured human-written text may sometimes be classified as AI-generated.

## Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Place the `AI_Human.csv` dataset file in the project folder before running `logic.py`.

Then run:

```bash
python logic.py
streamlit run app.py
```

## Note

This project is for educational purposes only. The prediction should not be treated as final proof that a text was written by AI or a human.
