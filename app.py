

from blockchain import Blockchain
blockchain = Blockchain()
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import hashlib
import json
import time
import nltk
import os
import pickle
import numpy as np
import pandas as pd
import re
import cv2
from PIL import Image
import easyocr

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import csv

app = Flask(__name__)
app.secret_key = "secret123"
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    review = db.Column(db.Text, nullable=False)
    result = db.Column(db.Integer, nullable=False)  # 0: Genuine, 1: Fake
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    total_fake = db.Column(db.Integer, default=0)
    total_genuine = db.Column(db.Integer, default=0)
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(previous_hash='0')

    def create_block(self, data=None, previous_hash=''):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(time.time()),
            'data': data,
            'previous_hash': previous_hash
        }

        block['hash'] = self.hash(block)
        self.chain.append(block)
        return block

    def add_review(self, data):
        previous_hash = self.chain[-1]['hash']
        self.create_block(data, previous_hash)

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
# Create blockchain object
blockchain = Blockchain()
# Load model and tokenizer
MAX_WORDS = 5000
MAX_LEN = 200
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

model = None
tokenizer = None

def load_ml_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        try:
            print("Loading ML model...")
            model_path = os.path.join(BASE_DIR, 'models', 'fake_review_model.h5')
            tokenizer_path = os.path.join(BASE_DIR, 'models', 'tokenizer.pkl')

            from tensorflow.keras.models import load_model
            import pickle

            model = load_model(model_path, compile=False)
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)

            print("Model loaded successfully")

        except Exception as e:
            print("Model loading failed:", e)
            model = None
            tokenizer = None
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = wordpunct_tokenize(text)
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)

def predict_review(text):

    # Rule-based detection
    if len(text.split()) < 3:
        return 1, 0.9
    
    if re.search(r'(http|www|buy now|click here|free|offer)', text.lower()):
        return 1, 0.95

    load_ml_model()

    if model is None or tokenizer is None:
        return 0, 0.5

    clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    try:
        prob_fake = float(model.predict(pad)[0][0])
    except:
        return 0, 0.5

    if prob_fake > 0.5:
        return 1, prob_fake
    else:
        return 0, 1 - prob_fake
    # Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))
@app.route("/health")
def health():
    return "OK"
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        print("Trying login with:", username)

        user = User.query.filter_by(username=username).first()

        print("User found:", user)

        if user:
            print("Stored hash:", user.password)
            print("Password check:", check_password_hash(user.password, password))

        if user and check_password_hash(user.password, password):
            login_user(user)
            print("Login success")
            return redirect(url_for('dashboard'))
        else:
            print("Login failed")
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('login.html', error='Passwords do not match')

        if User.query.filter_by(username=username).first():
            return render_template('login.html', error='Username already exists')

        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash("Account created successfully!")
        login_user(new_user)
        return redirect(url_for('login'))

    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    analyses = Analysis.query.filter_by(user_id=current_user.id).all()
    
    # Calculate statistics
    total_analyses = len(analyses)
    total_fake = sum(1 for a in analyses if a.result == 1)
    total_genuine = sum(1 for a in analyses if a.result == 0)
    
    return render_template('dashboard.html', 
                         analyses=analyses,
                         total_analyses=total_analyses,
                         total_fake=total_fake,
                         total_genuine=total_genuine)
@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/upload_csv', methods=['POST'])
@login_required
def upload_csv():
    file = request.files['file']

    if not file:
        return "No file uploaded"

    import pandas as pd
    df = pd.read_csv(file)

    results = []

    for review in df.iloc[:, 0]:
        review = str(review)

        if review.strip() != "" and review != "nan":
            prediction, confidence = predict_review(review)

            # Add to blockchain if Genuine
            if prediction == 0:
                blockchain.add_review({
                    "review": review,
                    "confidence": round(confidence * 100, 2),
                    "user": current_user.username,
                    "source": "CSV"
                })

            results.append((review, prediction, confidence))

    return render_template('image_results.html', results=results)
@app.route('/upload_image', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        try:
            file = request.files['image']

            if not file:
                return "No file uploaded"

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            print("Saved file at:", filepath)

            # Use PIL instead of cv2 (more reliable on Render)
            from PIL import Image
            import pytesseract

            try:
                img = Image.open(filepath)
                reader = easyocr.Reader(['en'])

                result = reader.readtext(filepath, detail=0)
                text = " ".join(result)
            except Exception as e:
                print("OCR Error:", e)
                return "OCR failed: " + str(e)

            reviews = text.split('\n')

            results = []
            for review in reviews:
                if review.strip() != "":
                    prediction, confidence = predict_review(review)

                    if prediction == 0:
                        blockchain.add_review({
                            "review": review,
                            "confidence": round(confidence * 100, 2),
                            "user": current_user.username,
                            "source": "Screenshot"
                        })

                    results.append((review, prediction, confidence))

            return render_template('image_results.html', results=results)

        except Exception as e:
            import traceback
            return "<pre>" + traceback.format_exc() + "</pre>"

    return render_template('upload_image.html')
@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html')

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        review = request.form.get('review', '').strip()

        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400

        prediction, confidence = predict_review(review)

        # Get current stats
        analyses = Analysis.query.filter_by(user_id=current_user.id).all()
        total_fake = sum(1 for a in analyses if a.result == 1)
        total_genuine = sum(1 for a in analyses if a.result == 0)

        # Save analysis
        analysis = Analysis(
            user_id=current_user.id,
            review=review,
            result=prediction,
            confidence=confidence,
            total_fake=total_fake + (1 if prediction == 1 else 0),
            total_genuine=total_genuine + (1 if prediction == 0 else 0)
        )

        db.session.add(analysis)
        db.session.commit()

        # -------- ADD BLOCKCHAIN HERE --------
        if prediction == 0:
            blockchain.add_review({
                "review": review,
                "confidence": round(confidence * 100, 2),
                "user": current_user.username,
                "source": "Manual"
            })
        # -------------------------------------

        return jsonify({
            'result': 'Fake' if prediction == 1 else 'Genuine',
            'confidence': round(confidence * 100, 2),
            'total_fake': analysis.total_fake,
            'total_genuine': analysis.total_genuine,
            'analysis_id': analysis.id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            return jsonify({'error': 'Only CSV and Excel files are supported'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        results = []
        
        # Analyze each review
        for idx, row in df.iterrows():
            review_text = str(row.iloc[0])  # Get first column
            prediction, confidence = predict_review(review_text)
            
            analysis = Analysis(
                user_id=current_user.id,
                review=review_text,
                result=prediction,
                confidence=confidence
            )
            db.session.add(analysis)
            
            results.append({
                'review': review_text[:100] + '...' if len(review_text) > 100 else review_text,
                'result': 'Fake' if prediction == 1 else 'Genuine',
                'confidence': round(confidence, 2)
            })
        
        db.session.commit()
        
        # Get updated stats
        analyses = Analysis.query.filter_by(user_id=current_user.id).all()
        total_fake = sum(1 for a in analyses if a.result == 1)
        total_genuine = sum(1 for a in analyses if a.result == 0)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_fake': total_fake,
            'total_genuine': total_genuine
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<int:analysis_id>')
@login_required
def results(analysis_id):
    analysis = Analysis.query.get(analysis_id)

    if not analysis or analysis.user_id != current_user.id:
        return redirect(url_for('dashboard'))

    # Get all user analyses
    all_analyses = Analysis.query.filter_by(user_id=current_user.id).all()

    total_fake = sum(1 for a in all_analyses if a.result == 1)
    total_genuine = sum(1 for a in all_analyses if a.result == 0)

    return render_template(
        'results.html',
        analysis=analysis,
        total_fake=total_fake,
        total_genuine=total_genuine,
        total_reviews=len(all_analyses),
        history=all_analyses
    )
@app.route('/show_users')
def show_users():
    users = User.query.all()
    return "<br>".join([u.username for u in users])
@app.route('/users_table')
def users_table():
    users = User.query.all()
    table = "<h2>User Table</h2><table border=1><tr><th>ID</th><th>Username</th><th>Password</th></tr>"
    
    for u in users:
        table += f"<tr><td>{u.id}</td><td>{u.username}</td><td>{u.password}</td></tr>"
    
    table += "</table>"
    return table

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
@app.route('/blockchain_table')
@login_required
def blockchain_table():
    return render_template('blockchain_table.html', chain=blockchain.chain)
@app.route('/test_model')
def test_model():
    test_reviews = [
        "Worst product ever waste of money",
        "Very bad quality broke in one day",
        "Amazing product loved it",
        "Highly recommend this product"
    ]

    results = []

    for review in test_reviews:
        clean = preprocess_text(review)
        seq = tokenizer.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=MAX_LEN)

        prob = float(model.predict(pad)[0][0])
        results.append((review, prob))

    return str(results)
# Defaultly adding user name and password
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)