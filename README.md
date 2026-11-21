# Email Spam Detection

Predicts whether an email is spam from simple numeric features, with a Streamlit web app.

## Project layout

| File | Purpose |
|---|---|
| `data/email_spam_detection.csv` | Dataset (7,000 emails, 30.5% spam) |
| `EmailSpamDetection.ipynb` | Exploration, cleaning, model comparison |
| `train.py` | Trains the models and saves `spam_model.joblib` |
| `spam_utils.py` | Feature list + preprocessing shared by training and the app |
| `email_spam_detection.py` | Streamlit app |
| `spam_model.joblib` | Trained pipeline (cleaning + scaling + Random Forest) |
| `requirements.txt` | Dependencies |

## Run it

```bash
pip install -r requirements.txt
python train.py                           # optional: retrain
streamlit run email_spam_detection.py
```

## Data findings

- **Features used:** `Email_Length`, `Num_Links`, `Num_Special_Chars`, `Capital_Words`, `Has_Attachment`.
- **Dropped:** `Email_ID` (identifier), `Subject` and `Sender_Email` domain. Spam rate is ~30% in every subject and domain, so they carry no signal.
- **Cleaning (done inside the saved pipeline, so the app applies it too):** 41 missing `Email_Length` values and 23 impossible negative `Num_Special_Chars` values become missing, then are filled with the median.
- **Outliers:** `Num_Links` goes up to 99; the models are not sensitive to this.

## Results

Hold-out test set (20%): accuracy 99.9%, F1 0.999, ROC AUC 1.0 (Random Forest, best of Logistic Regression / Random Forest / Gradient Boosting by 5-fold CV F1).

These scores are unusually high because this dataset looks synthetic and spam is almost perfectly separable by length, links, special characters and capital words. Expect lower performance on real-world email, which would need text features (body/subject text) to do well.
