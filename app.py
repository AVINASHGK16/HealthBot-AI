import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import joblib
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- DATABASE MODELS ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    sessions = db.relationship('ChatSession', backref='owner', lazy=True)

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), default="New Consultation")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default="started") 
    collected_symptoms = db.Column(db.Text, default="[]") 
    messages = db.relationship('ChatMessage', backref='session', cascade="all, delete-orphan")

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    sender = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    options = db.Column(db.String(200), nullable=True) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ML CONFIGURATION ---
DATASET_PATH = 'Training.csv'  
MODEL_PATH = 'disease_model.pkl'
COLUMNS_PATH = 'symptom_columns.pkl'

def train_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            X = df.iloc[:, :-1] 
            y = df.iloc[:, -1]  
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            joblib.dump(X.columns.tolist(), COLUMNS_PATH)
            print("Model Trained.")
        except Exception as e:
            print(f"Error: {e}")

train_model_if_needed()

# --- INTELLIGENCE ENGINE ---

def extract_symptoms(user_text):
    if not os.path.exists(COLUMNS_PATH): return None, []
    symptom_columns = joblib.load(COLUMNS_PATH)
    user_text = user_text.lower()
    
    symptom_map = {
        'high_fever': ['fever', 'high temp', 'hot', 'temperature', 'burning up'],
        'mild_fever': ['mild fever', 'low fever', 'slightly warm'],
        'chills': ['chills', 'shivering', 'cold', 'shaking', 'freezing'],
        'fatigue': ['tired', 'exhausted', 'drained', 'fatigued', 'lethargic', 'low energy', 'weak'],
        'malaise': ['unwell', 'feeling bad', 'sick', 'body feels off'],
        'muscle_pain': ['muscle pain', 'body ache', 'sore muscles', 'pain in body', 'aches'],
        'sweating': ['sweat', 'perspiring', 'clammy', 'soaked', 'night sweats'],
        'headache': ['head', 'migraine', 'head hurts', 'throbbing', 'pounding head'],
        'dizziness': ['dizzy', 'spin', 'lightheaded', 'faint', 'woozy'],
        'altered_sensorium': ['confused', 'drowsy', 'lost senses', 'disoriented', 'brain fog'],
        'stomach_pain': ['stomach', 'belly', 'pain in tummy', 'abdominal', 'gut pain', 'cramps'],
        'acidity': ['acid', 'heartburn', 'sour', 'reflux', 'burning stomach'],
        'vomiting': ['vomit', 'puke', 'throw up', 'nausea', 'queasy', 'retching'],
        'diarrhoea': ['diarrhea', 'loose motion', 'runny poop', 'the runs', 'watery stool'],
        'cough': ['cough', 'coughing', 'dry cough'],
        'breathlessness': ['breath', 'breathing', 'short of breath', 'cant breathe', 'gasping', 'panting'],
        'chest_pain': ['chest', 'heart', 'ribs', 'tightness in chest', 'heart pain', 'pressure on chest'],
        'throat_irritation': ['sore throat', 'itchy throat', 'throat hurts', 'scratchy throat'],
        'runny_nose': ['runny nose', 'sniffles', 'watery nose'],
        'congestion': ['congestion', 'stuffy', 'blocked nose', 'nose block'],
        'continuous_sneezing': ['sneeze', 'sneezing', 'keep sneezing'],
        'itching': ['itch', 'scratch', 'itchy skin'],
        'skin_rash': ['rash', 'spots', 'redness', 'hives', 'bumps', 'breakout'],
        'yellowish_skin': ['yellow skin', 'jaundice', 'pale', 'skin is yellow'],
        'joint_pain': ['joint', 'knees', 'elbows', 'wrists', 'ankles', 'aching joints'],
        'muscle_weakness': ['weak muscles', 'hard to lift', 'weakness', 'cant lift'],
        'neck_pain': ['neck', 'stiff neck', 'cant turn head'],
        'weakness_of_one_body_side': ['paralysis', 'cant move one side', 'numbness on one side', 'stroke', 'one side weak', 'drooping face'],
        'weight_loss': ['lost weight', 'weight loss', 'skinny', 'thinning'], 
        'weight_gain': ['gained weight', 'fat'],
        'excessive_hunger': ['hungry', 'starving', 'eat a lot', 'always hungry'],
        'polyuria': ['peeing a lot', 'lots of urine']
    }
    
    detected = []
    for col, keywords in symptom_map.items():
        for k in keywords:
            if k in user_text:
                if col in symptom_columns:
                    detected.append(col)
                    break
    return detected

def get_next_question(current_symptoms):
    if 'chest_pain' in current_symptoms and 'breathlessness' not in current_symptoms and 'sweating' not in current_symptoms:
        return ("Chest pain can be serious. Are you experiencing difficulty breathing or sweating?", "Yes Difficulty Breathing,Yes Sweating,No just Pain", "clarifying_chest")
    if 'high_fever' in current_symptoms and 'chills' in current_symptoms and 'muscle_pain' not in current_symptoms:
        return ("With fever and chills, do you also have body aches or excessive sweating?", "Yes Body Ache,Yes Sweating,No", "clarifying_malaria")
    if 'polyuria' in current_symptoms or 'excessive_hunger' in current_symptoms:
        return ("Have you noticed sudden weight loss or blurry vision?", "Yes Weight Loss,Yes Blurry Vision,No", "clarifying_diabetes")
    if 'headache' in current_symptoms and 'migraine' not in current_symptoms:
        return ("To help me pinpoint the cause, where exactly is the pain?", "Forehead (Sinus),One Side (Migraine),Back of Head,All Over", "clarifying_headache")
    if 'high_fever' in current_symptoms and 'skin_rash' not in current_symptoms:
        return ("Are you also shivering or do you see a rash?", "Yes Shivering,No,I have a Rash too", "clarifying_fever")
    return None

def get_disease_details(disease_name):
    """ Returns a dictionary with detailed educational info about the disease. """
    
    # Generic Fallback
    default = {
        "desc": "A medical condition affecting the body's normal functions.",
        "causes": "Various factors including infections, genetics, or lifestyle.",
        "risk": "Complications may arise if left untreated.",
        "action": "Consult a general physician for a proper diagnosis.",
        "link": f"https://www.google.com/search?q={disease_name}+symptoms+treatment"
    }

    db = {
        'Fungal infection': {
            "desc": "A skin infection caused by a fungus.",
            "causes": "Moisture trapped in skin folds, weak immune system, or contact with infected surfaces.",
            "risk": "Can spread to other body parts or cause secondary bacterial infections.",
            "action": "Keep the area dry. Use antifungal creams.",
            "link": "https://www.healthline.com/health/fungal-infection"
        },
        'Allergy': {
            "desc": "An immune system reaction to a foreign substance.",
            "causes": "Pollen, pet dander, dust mites, or certain foods.",
            "risk": "Can lead to sinus infections or severe anaphylaxis in rare cases.",
            "action": "Identify triggers. Take antihistamines.",
            "link": "https://www.mayoclinic.org/diseases-conditions/allergies/symptoms-causes/syc-20351497"
        },
        'GERD': {
            "desc": "Gastroesophageal Reflux Disease (Acid Reflux).",
            "causes": "Stomach acid flowing back into the tube connecting your mouth and stomach.",
            "risk": "Esophageal damage, dental problems, and chronic cough.",
            "action": "Avoid spicy food/caffeine. Don't lie down immediately after eating.",
            "link": "https://www.webmd.com/heartburn-gerd/guide/reflux-disease-gerd-1"
        },
        'Heart attack': {
            "desc": "A blockage of blood flow to the heart muscle.",
            "causes": "Blocked arteries (coronary artery disease), blood clots.",
            "risk": "Permanent heart damage, heart failure, or death.",
            "action": "EMERGENCY: Call ambulance. Chew aspirin immediately.",
            "link": "https://www.heart.org/en/health-topics/heart-attack"
        },
        'Migraine': {
            "desc": "A headache of varying intensity, often accompanied by nausea and sensitivity to light.",
            "causes": "Hormonal changes, stress, drinks (alcohol/caffeine), sensory stimuli.",
            "risk": "Chronic daily headaches, status migrainosus.",
            "action": "Rest in a dark, quiet room. Apply a cold compress.",
            "link": "https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201"
        },
        'Malaria': {
            "desc": "A disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes.",
            "causes": "Bite of an infected Anopheles mosquito.",
            "risk": "Kidney failure, seizures, mental confusion, coma, or death if untreated.",
            "action": "Consult a doctor immediately for blood tests. Use mosquito nets.",
            "link": "https://www.cdc.gov/malaria/about/disease.html"
        },
        'Jaundice': {
            "desc": "A condition causing yellowing of the skin and eyes.",
            "causes": "Excess bilirubin, hepatitis, gallstones, or tumors.",
            "risk": "Liver failure, bleeding disorders, or kidney failure.",
            "action": "Rest completely. Eat boiled, oil-free food. Drink sugarcane juice.",
            "link": "https://www.nhs.uk/conditions/jaundice/"
        },
        'Typhoid': {
            "desc": "A bacterial infection that can lead to a high fever, diarrhea, and vomiting.",
            "causes": "Salmonella typhi bacteria via contaminated food or water.",
            "risk": "Intestinal bleeding or holes (perforation), which can be fatal.",
            "action": "Antibiotics are required. Drink only boiled water.",
            "link": "https://www.mayoclinic.org/diseases-conditions/typhoid-fever/symptoms-causes/syc-20378661"
        },
        'Common Cold': {
            "desc": "A viral infection of your nose and throat (upper respiratory tract).",
            "causes": "Viruses (rhinoviruses) spread through air or contact.",
            "risk": "Ear infections, asthma attacks, or sinusitis.",
            "action": "Rest, hydration, and steam inhalation.",
            "link": "https://www.mayoclinic.org/diseases-conditions/common-cold/symptoms-causes/syc-20351605"
        },
        'Paralysis (brain hemorrhage)': {
            "desc": "Loss of muscle function in part of your body.",
            "causes": "Stroke, spinal cord injury, or nerve damage.",
            "risk": "Permanent disability, difficulty speaking or swallowing.",
            "action": "EMERGENCY: Seek hospital admission immediately.",
            "link": "https://medlineplus.gov/paralysis.html"
        }
    }
    
    return db.get(disease_name.strip(), default)

def check_critical_rules(detected_symptoms):
    if 'chest_pain' in detected_symptoms:
        if 'breathlessness' in detected_symptoms or 'sweating' in detected_symptoms:
            return "Heart attack"
        return "Potential Heart Risk"
    if 'weakness_of_one_body_side' in detected_symptoms or 'altered_sensorium' in detected_symptoms:
        return "Paralysis (brain hemorrhage)"
    return None

def apply_safety_check(prediction, detected_symptoms):
    if prediction == 'Paralysis (brain hemorrhage)':
        if not any(x in detected_symptoms for x in ['weakness_of_one_body_side', 'altered_sensorium']):
            if 'headache' in detected_symptoms: return "Migraine" 
            return "Viral Fever" 
    if prediction == 'Heart attack':
        if 'chest_pain' not in detected_symptoms: return "Gastritis or Anxiety"
    return prediction

# --- ROUTES ---

@app.route('/')
@login_required
def home():
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).all()
    return render_template('index.html', user_name=current_user.username, sessions=sessions)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username taken.')
            return redirect(url_for('register'))
        new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/new_chat', methods=['POST'])
@login_required
def new_chat():
    new_session = ChatSession(user_id=current_user.id, title="New Consultation", status="started", collected_symptoms="[]")
    db.session.add(new_session)
    db.session.commit()
    
    user_name = current_user.username.capitalize()
    greetings = [f"Hello {user_name}, I'm Dr. Bot. How can I help you today?", f"Hi {user_name}. What symptoms are bothering you?"]
    welcome_text = random.choice(greetings)
    msg = ChatMessage(session_id=new_session.id, sender='bot', content=welcome_text)
    db.session.add(msg)
    db.session.commit()
    return jsonify({"session_id": new_session.id})

@app.route('/delete_chat/<int:session_id>', methods=['DELETE'])
@login_required
def delete_chat(session_id):
    session = ChatSession.query.get_or_404(session_id)
    if session.user_id != current_user.id: return jsonify({"error": "Unauthorized"}), 403
    db.session.delete(session)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/get_chat_history/<int:session_id>')
@login_required
def get_chat_history(session_id):
    session = ChatSession.query.get_or_404(session_id)
    if session.user_id != current_user.id: return jsonify({"error": "Unauthorized"}), 403
    messages = [{"sender": msg.sender, "content": msg.content, "options": msg.options.split(',') if msg.options else None} for msg in session.messages]
    return jsonify({"messages": messages})

@app.route('/download_report/<int:session_id>')
@login_required
def download_report(session_id):
    session = ChatSession.query.get_or_404(session_id)
    if session.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="HealthBot AI - Consultation Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient: {current_user.username}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnosis Result:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"{session.title}", ln=True)
    pdf.ln(5)
    symptoms = json.loads(session.collected_symptoms)
    formatted_symptoms = [s.replace('_', ' ').title() for s in symptoms]
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Reported Symptoms:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=", ".join(formatted_symptoms) if formatted_symptoms else "No specific symptoms recorded.")
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 10, txt="Disclaimer: AI-generated report. Consult a doctor.")
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=HealthBot_Report_{session.id}.pdf'
    return response

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    user_text = data.get('message', '')
    session_id = data.get('session_id')
    
    chat_session = ChatSession.query.get(session_id)
    if not chat_session:
        return jsonify({'error': 'Session not found'}), 404

    # 1. Save User Message
    user_msg = ChatMessage(session_id=session_id, sender='user', content=user_text)
    db.session.add(user_msg)
    
    # 2. Extract Symptoms
    current_symptoms = json.loads(chat_session.collected_symptoms)
    new_symptoms = extract_symptoms(user_text)
    
    if new_symptoms:
        for s in new_symptoms:
            if s not in current_symptoms:
                current_symptoms.append(s)
        chat_session.collected_symptoms = json.dumps(current_symptoms)

    # 3. Decision Logic
    bot_response = ""
    options = []
    
    # If no symptoms found yet
    if not current_symptoms:
        bot_response = "I couldn't detect specific symptoms in that message. Could you describe your symptoms differently? (e.g., 'I have a headache and fever')"
    
    else:
        # Check for clarifying questions
        next_q = get_next_question(current_symptoms)
        
        if next_q and chat_session.status != 'diagnosed':
            question, opts, tag = next_q
            # Check if we already asked this (simple check: implies we need state tracking, 
            # for now, we assume if symptoms are present, we skip. 
            # In a pro app, we'd track 'questions_asked' list.)
            bot_response = question
            options = opts.split(',')
        
        else:
            # --- MAKE PREDICTION ---
            if not os.path.exists(MODEL_PATH):
                bot_response = "System is initializing. Please try again in 10 seconds."
            else:
                model = joblib.load(MODEL_PATH)
                columns = joblib.load(COLUMNS_PATH)
                
                # Prepare vector
                input_vector = np.zeros(len(columns))
                for symptom in current_symptoms:
                    if symptom in columns:
                        idx = columns.index(symptom)
                        input_vector[idx] = 1
                
                prediction = model.predict([input_vector])[0]
                
                # Safety Checks
                prediction = apply_safety_check(prediction, current_symptoms)
                
                # Get Educational Info
                info = get_disease_details(prediction)
                
                # --- GENERATE THE CARD WITH DISCLAIMER ---
                bot_response = f"""
                <div class="bg-teal-50 dark:bg-slate-700/50 p-4 rounded-xl border border-teal-100 dark:border-slate-600 mb-2">
                    <div class="flex items-center gap-3 mb-3 border-b border-teal-200 dark:border-slate-600 pb-2">
                        <div class="w-10 h-10 bg-teal-100 dark:bg-teal-900 rounded-full flex items-center justify-center text-teal-600 dark:text-teal-400">
                            <i class="fa-solid fa-user-doctor text-lg"></i>
                        </div>
                        <div>
                            <p class="text-xs font-bold text-gray-400 uppercase tracking-wider">Analysis Result</p>
                            <h3 class="text-lg font-bold text-gray-800 dark:text-white">{prediction}</h3>
                        </div>
                    </div>
                    
                    <div class="space-y-3 text-sm">
                        <div>
                            <p class="font-semibold text-gray-500 dark:text-gray-400 text-xs mb-1"><i class="fa-solid fa-circle-info"></i> ABOUT</p>
                            <p class="text-gray-700 dark:text-gray-300 leading-relaxed">{info['desc']}</p>
                        </div>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                            <div class="bg-white dark:bg-slate-800 p-3 rounded-lg">
                                <p class="font-bold text-gray-500 text-xs mb-1">CAUSES</p>
                                <p class="text-gray-600 dark:text-gray-400 text-xs">{info['causes']}</p>
                            </div>
                            <div class="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg">
                                <p class="font-bold text-red-500 text-xs mb-1">RISK IF IGNORED</p>
                                <p class="text-red-600 dark:text-red-300 text-xs">{info['risk']}</p>
                            </div>
                        </div>

                        <div class="bg-teal-100/50 dark:bg-teal-900/30 p-3 rounded-lg border border-teal-200 dark:border-teal-800/50">
                            <p class="font-bold text-teal-700 dark:text-teal-400 text-xs mb-1"><i class="fa-solid fa-notes-medical"></i> IMMEDIATE ACTION</p>
                            <p class="text-gray-700 dark:text-gray-300 font-medium">{info['action']}</p>
                        </div>
                    </div>

                    <div class="mt-4 pt-3 border-t border-gray-200 dark:border-gray-600 text-center">
                        <p class="text-[10px] text-gray-400 italic">
                            <i class="fa-solid fa-triangle-exclamation"></i> 
                            AI Prediction is not 100% accurate. This is not a substitute for professional medical advice. 
                            <span class="block mt-1">Please consult a real doctor.</span>
                        </p>
                    </div>

                    <div class="mt-4 flex gap-2">
                        <a href="https://www.google.com/search?q=doctors+near+me" target="_blank" class="flex-1 bg-teal-600 hover:bg-teal-700 text-white text-center py-2 rounded-lg text-sm font-semibold transition shadow-sm">
                            <i class="fa-solid fa-user-doctor mr-1"></i> Find Doctor
                        </a>
                        <a href="{info['link']}" target="_blank" class="flex-1 bg-[#1e293b] hover:bg-black text-white text-center py-2 rounded-lg text-sm font-semibold transition shadow-sm">
                            <i class="fa-solid fa-book-medical mr-1"></i> Read More
                        </a>
                    </div>
                </div>
                """
                chat_session.status = 'diagnosed'
                # Generate a title if it's the first diagnosis
                if chat_session.title == "New Consultation":
                    chat_session.title = f"Consultation: {prediction}"

    # 4. Save Bot Message
    bot_msg = ChatMessage(session_id=session_id, sender='bot', content=bot_response, options=",".join(options) if options else None)
    db.session.add(bot_msg)
    db.session.commit()
    
    return jsonify({
        'response': bot_response, 
        'options': options,
        'new_title': chat_session.title if chat_session.status == 'diagnosed' else None
    })


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("Starting Server...")
    app.run(debug=True, port=5000)