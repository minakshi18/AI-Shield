from flask import Flask, render_template, request, jsonify
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
from PIL import Image
import pytesseract

# --- NLTK Setup for Vercel ---
nltk.data.path.append("/tmp")
nltk.download('stopwords', download_dir="/tmp")

app = Flask(__name__)

# --- Load Models with Correct Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
tfidf_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
tfidf = pickle.load(open(tfidf_path, 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def predict_cleaning(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [ps.stem(word) for word in text if not word in stop_words]
    return " ".join(text)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    raw_text = ""
    if 'news_image' in request.files and request.files['news_image'].filename != '':
        img = Image.open(request.files['news_image'])
        raw_text = pytesseract.image_to_string(img)
    else:
        raw_text = request.form.get('news_text', '')

    if len(raw_text.split()) < 5:
        return render_template('index.html', error="⚠️ Content too short to analyze.", original_text=raw_text)

    cleaned_text = predict_cleaning(raw_text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    
    decision_score = model.decision_function(vectorized_text)[0]
    confidence = 1 / (1 + np.exp(-decision_score))
    
    prob_real = round((1 - confidence) * 100, 2) if prediction == 0 else round((1 - confidence) * 100, 2)
    prob_fake = 100 - prob_real
    
    status, color = "", ""
    display_prob = prob_real if prediction == 0 else prob_fake
    
    if prediction == 0:
        if display_prob >= 95: status, color = "VERIFIED REAL ✅", "#22C55E"
        elif 85 <= display_prob <= 94: status, color = "CLOSE TO REAL 🟢", "#84cc16"
        elif 70 <= display_prob <= 84: status, color = "MIGHT BE ONLY 1% CHANCE THAT NEWS IS FAKE 🔍", "#eab308"
        else: status, color = "UNCERTAIN / LIKELY FAKE ⚠️", "#EF4444"
    else:
        status, color = "FAKE NEWS DETECTED ⚠️", "#EF4444"

    return render_template('index.html', prediction=prediction, prob_real=prob_real, 
                           prob_fake=prob_fake, status=status, color=color, original_text=raw_text)

# --- Vercel Specific ---
app = app

if __name__ == '__main__':
    app.run(debug=True)
