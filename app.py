from flask import Flask, request, jsonify
import pandas as pd
import re
from pyvi import ViTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

app = Flask(__name__, template_folder='docs')

# Load data and preprocess it
df = pd.read_csv('comments_data.csv')
df.dropna(inplace=True)
df.drop(columns=['title', 'id'], inplace=True)

def standardize_sentence(text):
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def word_separation(text):
    text = ViTokenizer.tokenize(text)
    return text

def preprocess(text):
    text = word_separation(text)
    text = standardize_sentence(text)
    return text

df['content'] = df['content'].apply(preprocess)

# Define sentiment from rating
def classify_rating(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['rating'].apply(classify_rating)

# Prepare data for modeling
X = df['content']
y = df['sentiment']
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)
X_Train, X_test, y_Train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Oversample using SMOTE
smote = SMOTE(random_state=42)
X_Train_resampled, y_Train_resampled = smote.fit_resample(X_Train, y_Train)

# Train logistic regression model
model_lr = LogisticRegression(C=100, penalty='l1', solver='liblinear')
best_model_lr = model_lr.fit(X_Train_resampled, y_Train_resampled)
y_pred_lr = best_model_lr.predict(X_test)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        comment = data['comment']
        processed_comment = preprocess(comment)
        sentiment = best_model_lr.predict(tfidf_vectorizer.transform([processed_comment]))[0]
        accuracy = accuracy_score(y_test, y_pred_lr)
        return jsonify({'sentiment': sentiment, 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
