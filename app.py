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
    try:
        data = request.json
        user_message = data['message']
        session_id = data.get('session_id')
        session = ChatSession.query.get(session_id)
        if not session: return jsonify({"error": "Invalid Session"})

        user_msg_db = ChatMessage(session_id=session.id, sender='user', content=user_message)
        db.session.add(user_msg_db)
        
        current_symptoms = json.loads(session.collected_symptoms)
        new_symptoms = extract_symptoms(user_message)
        
        if session.status == "clarifying_malaria":
            if "sweat" in user_message.lower(): current_symptoms.append("sweating")
            if "ache" in user_message.lower(): current_symptoms.append("muscle_pain")
            session.status = "active"
        elif session.status == "clarifying_diabetes":
            if "weight" in user_message.lower(): current_symptoms.append("weight_loss")
            if "vision" in user_message.lower() or "blur" in user_message.lower(): current_symptoms.append("blurred_and_distorted_vision")
            session.status = "active"
        elif session.status == "clarifying_chest":
            if "breath" in user_message.lower(): current_symptoms.append("breathlessness")
            if "sweat" in user_message.lower(): current_symptoms.append("sweating")
            session.status = "active"
        elif session.status == "clarifying_headache":
            if "one side" in user_message.lower(): current_symptoms.append("migraine")
            session.status = "active"
        elif session.status == "clarifying_fever":
            if "shivering" in user_message.lower(): current_symptoms.append("chills")
            if "rash" in user_message.lower(): current_symptoms.append("skin_rash")
            session.status = "active"

        all_symptoms = list(set(current_symptoms + new_symptoms))
        session.collected_symptoms = json.dumps(all_symptoms)
        db.session.add(session)
        db.session.commit()
        
        next_step = get_next_question(all_symptoms)
        if next_step and session.status == "started": 
            question, options_str, next_state = next_step
            session.status = next_state 
            bot_msg_db = ChatMessage(session_id=session.id, sender='bot', content=question, options=options_str)
            db.session.add(bot_msg_db)
            db.session.commit()
            return jsonify({"response": question, "options": options_str.split(',')})

        if not all_symptoms:
            user_name = current_user.username.capitalize()
            response_text = f"I'm listening, {user_name}. Could you describe your symptoms? (e.g., 'stomach pain', 'dizzy')"
            options = None
        else:
            critical_diagnosis = check_critical_rules(all_symptoms)
            if critical_diagnosis:
                final_pred = critical_diagnosis
            else:
                symptom_columns = joblib.load(COLUMNS_PATH)
                input_vector = np.zeros(len(symptom_columns))
                for sym in all_symptoms:
                    if sym in symptom_columns:
                        input_vector[symptom_columns.index(sym)] = 1
                model = joblib.load(MODEL_PATH)
                raw_pred = model.predict([input_vector])[0]
                final_pred = apply_safety_check(raw_pred, all_symptoms)

            # GET RICH DETAILS
            info = get_disease_details(final_pred)
            
            if session.title == "New Consultation":
                session.title = f"{final_pred}"
                db.session.add(session)

            specialist_map = {
                'Heart': 'Cardiologist', 'Chest': 'Cardiologist',
                'Skin': 'Dermatologist', 'Rash': 'Dermatologist', 'Acne': 'Dermatologist', 'Psoriasis': 'Dermatologist',
                'Stomach': 'Gastroenterologist', 'Acidity': 'Gastroenterologist', 'Ulcer': 'Gastroenterologist', 'Vomit': 'Gastroenterologist',
                'Joint': 'Orthopedic', 'Knee': 'Orthopedic', 'Bone': 'Orthopedic', 'Arthritis': 'Orthopedic',
                'Eye': 'Ophthalmologist', 'Vision': 'Ophthalmologist',
                'Diabetes': 'Endocrinologist', 'Sugar': 'Endocrinologist', 'Thyroid': 'Endocrinologist',
                'Malaria': 'General Physician', 'Typhoid': 'General Physician', 'Fever': 'General Physician', 'Cold': 'General Physician',
                'Brain': 'Neurologist', 'Paralysis': 'Neurologist', 'Migraine': 'Neurologist',
                'Urinary': 'Urologist'
            }
            specialist = "General Physician"
            for key, val in specialist_map.items():
                if key in final_pred:
                    specialist = val
                    break
            maps_link = f"https://www.google.com/maps/search/{specialist}+near+me"

            severity_bg = "bg-emerald-50"
            severity_border = "border-emerald-500"
            severity_text = "text-emerald-800"
            icon = "fa-user-doctor"
            icon_color = "text-emerald-600"

            if "Heart" in final_pred or "Paralysis" in final_pred or "Risk" in final_pred or "Emergency" in final_pred:
                severity_bg = "bg-red-50"
                severity_border = "border-red-500"
                severity_text = "text-red-800"
                icon = "fa-triangle-exclamation"
                icon_color = "text-red-600"

            formatted_symptoms = [s.replace('_', ' ').title() for s in all_symptoms]
            
            response_text = f"""
            <div class="bg-white p-5 rounded-2xl shadow-sm border border-gray-100 w-full max-w-md">
                <div class="flex items-center gap-3 mb-4 {severity_bg} p-3 rounded-xl border-l-4 {severity_border}">
                    <div class="w-10 h-10 bg-white rounded-full flex items-center justify-center shadow-sm">
                        <i class="fa-solid {icon} {icon_color} text-lg"></i>
                    </div>
                    <div>
                        <p class="text-xs font-bold {severity_text} uppercase tracking-wider">Analysis Result</p>
                        <h3 class="text-lg font-bold text-gray-800 leading-tight">{final_pred}</h3>
                    </div>
                </div>
                
                <div class="space-y-4">
                    <div>
                        <p class="text-xs text-gray-400 font-semibold uppercase mb-1"><i class="fa-solid fa-circle-info mr-1"></i> About</p>
                        <p class="text-sm text-gray-700 leading-relaxed">{info['desc']}</p>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-gray-50 p-2 rounded-lg">
                            <p class="text-xs text-gray-500 font-bold uppercase mb-1">Causes</p>
                            <p class="text-xs text-gray-700">{info['causes']}</p>
                        </div>
                        <div class="bg-red-50 p-2 rounded-lg">
                            <p class="text-xs text-red-500 font-bold uppercase mb-1">Risk if Ignored</p>
                            <p class="text-xs text-gray-700">{info['risk']}</p>
                        </div>
                    </div>
                    
                    <div>
                        <p class="text-xs text-gray-400 font-semibold uppercase mb-1"><i class="fa-solid fa-user-nurse mr-1"></i> Immediate Action</p>
                        <p class="text-sm font-medium text-gray-800 leading-relaxed bg-teal-50 p-3 rounded-lg border border-teal-100">{info['action']}</p>
                    </div>

                    <div class="grid grid-cols-2 gap-2 pt-2">
                         <a href="{maps_link}" target="_blank" class="text-center bg-teal-600 hover:bg-teal-700 text-white font-medium py-2 rounded-lg transition-all shadow-md shadow-teal-200 text-xs flex items-center justify-center">
                            <i class="fa-solid fa-map-location-dot mr-1"></i> Find Doctor
                        </a>
                        <a href="{info['link']}" target="_blank" class="text-center bg-gray-800 hover:bg-gray-900 text-white font-medium py-2 rounded-lg transition-all shadow-md text-xs flex items-center justify-center">
                            <i class="fa-solid fa-book-medical mr-1"></i> Read More
                        </a>
                    </div>
                </div>

                <div class="mt-4 pt-3 border-t border-gray-100 flex justify-between items-center">
                    <a href="/download_report/{session.id}" class="text-[10px] text-gray-400 hover:text-gray-600 flex items-center gap-1"><i class="fa-solid fa-file-pdf"></i> Save Report</a>
                    <button onclick="createNewChat()" class="text-xs text-teal-600 font-bold hover:text-teal-700 flex items-center gap-1 transition">
                        New Checkup <i class="fa-solid fa-arrow-right"></i>
                    </button>
                </div>
            </div>
            """
            options = None
            session.status = "active"

        bot_msg_db = ChatMessage(session_id=session.id, sender='bot', content=response_text)
        db.session.add(bot_msg_db)
        db.session.commit()

        return jsonify({"response": response_text, "options": options, "new_title": session.title})

    except Exception as e:
        print(e)
        return jsonify({"response": "I'm having a little trouble thinking right now. Please try again."})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("Starting Server...")
    app.run(debug=True, port=5000)