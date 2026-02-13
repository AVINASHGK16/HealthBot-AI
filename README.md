🚀 AI Health Bot — Symptom-based disease prediction chatbot  
🛠 Tech: Python, Flask, HTML/CSS/JS, Random Forest  
🌐 Live Demo: https://huggingface.co/spaces/avinashgk/healthaibot 
📦 Features: Chatbot UI, disease prediction, login system

✨ Key Features
 Smart Symptom Extraction: Uses natural language keyword mapping to identify symptoms from conversational user input.

 ML Disease Prediction: Utilizes a custom-trained Random Forest model to predict potential diseases based on collected symptoms.

 Dynamic Clarification Engine: Automatically asks follow-up questions to distinguish between similar conditions (e.g., differentiating between a common cold and malaria).

 Downloadable PDF Reports: Generates professional, downloadable PDF summaries of the consultation using FPDF.
📱 Mobile-Optimized UI: Fully responsive design with native-feeling mobile scroll interactions and viewport adjustments.

 Secure User Authentication: Features a complete user registration and login system using Flask-Login and password hashing.

🛠️ Tech Stack
Backend: Python, Flask, Gunicorn

Machine Learning: Scikit-Learn (Random Forest), Pandas, NumPy, Joblib

Database: SQLite & Flask-SQLAlchemy

Frontend: HTML5, Tailwind CSS, FontAwesome Icons

Deployment: Docker, Hugging Face Spaces

🚀 Local Installation & Setup
If you want to run this project locally on your machine, follow these steps:

1. Clone the repository

Bash
git clone https://github.com/AVINASHGK16/HealthBot-AI.git
cd HealthBot-AI
2. Create a virtual environment

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies

Bash
pip install -r requirements.txt
4. Run the application
Note: Make sure Training.csv is in the root directory. The application is programmed to automatically train and generate the disease_model.pkl on its first run!

Bash
python app.py
5. Open your browser
Navigate to http://127.0.0.1:5000

⚠️ Disclaimer
HealthBot AI is an educational project and proof-of-concept. The AI predictions are not 100% accurate and should never be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition.
