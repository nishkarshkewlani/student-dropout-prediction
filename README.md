# student-dropout-prediction
# Student Dropout Prediction System

This is an end-to-end Machine Learning project that predicts whether a student is at risk of dropping out based on academic and personal attributes.  
The project integrates **Machine Learning**, **Flask backend**, and a **Web-based frontend UI**.

---

# Project Highlights
- Real-world education problem
- End-to-end ML pipeline
- Model served via REST API
- Web interface for real-time prediction
- Fully version-controlled on GitHub

---

# Problem Statement
Student dropout is a major issue in educational institutions.  
Early identification of at-risk students can help institutions take preventive actions.

This system predicts dropout risk using historical student data.

---

# Tech Stack
- **Programming Language:** Python
- **ML Libraries:** Pandas, NumPy, Scikit-learn
- **Backend:** Flask (REST API)
- **Frontend:** HTML, CSS, JavaScript
- **Tools:** Git, GitHub, GitHub Codespaces

---

# Project Structure
student-dropout-prediction/
│
├── backend/
│ └── app.py
├── ml/
│ ├── train_model.py
│ └── student_model.pkl
├── dataset/
│ └── student_data.csv
├── frontend/
│ └── index.html
├── requirements.txt
└── README.md

# How the System Works
1. Student enters academic details in the web UI
2. Frontend sends data to Flask API in JSON format
3. Flask loads the trained ML model
4. Model predicts dropout risk
5. Result is displayed on the UI

---

# How to Run the Project
```bash
pip install -r requirements.txt
python backend/app.py
Then open:
http://localhost:5000
```

# Sample Output

✅ Low Risk of Dropout
⚠️ High Risk of Dropout

# Future Improvements

Add database support
Improve feature engineering
Deploy on cloud (AWS / Render)
Add authentication for institutions

# Author

Nishkarsh Kewlani
B. Tech (CSE in Data Science)

