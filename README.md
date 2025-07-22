💰 AI Employee Income Predictor
A Streamlit web application that predicts an individual's approximate income based on various demographic and employment attributes. This tool leverages a pre-trained Machine Learning model (Random Forest Regressor) and provides comprehensive input validation and insightful visualizations.

✨ Live Demo
Experience the app live!
🔗 https://employee-salary-prediction-using-machine-learning-algorithm.streamlit.app/

Application Screenshot/GIF:
(Replace this placeholder with a GIF or screenshot of your deployed Streamlit app. You can drag and drop an image directly into your GitHub README editor, or upload it to your repository and link it here.)

🚀 Features
Intelligent Salary Prediction: Predicts approximate income based on user-provided inputs like age, experience, education, job title, and gender.

Comprehensive Input Validation: Robust checks for logical inconsistencies (e.g., age vs. experience, education vs. job title) with helpful error and warning messages.

Dynamic Market Insights: Provides predicted salary range, confidence level, expected market range for the role, and market position.

Interactive Visualizations:

Salary Distribution for a given job title.

Feature Importance in salary prediction.

(Optional: Requires adult 3.csv upload for detailed dataset insights) Average Income by Education Level, Occupation, and Gender.

Pre-trained Model Integration: Utilizes a pre-trained RandomForestRegressor model for fast predictions.

Clean & Intuitive UI: Built with Streamlit for an easy-to-use and responsive interface.

🛠️ Technologies Used
Python 3.x

Streamlit: For building the interactive web application.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Scikit-learn: For machine learning model (Random Forest Regressor) and preprocessing components.

Joblib: For saving and loading the pre-trained model.

Plotly: For creating interactive and visually appealing data visualizations.

⚙️ Setup & Installation (Local)
Follow these steps to get a local copy of the project up and running on your machine.

Prerequisites
Python 3.8+

pip (Python package installer)

git (Version control system)

Steps
Clone the repository:

git clone https://github.com/joelonney/Employee-salary-prediction-using-machine-learning-algorithm.git
cd Employee-salary-prediction-using-machine-learning-algorithm

Create a virtual environment (recommended):

python -m venv venv

Activate the virtual environment:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install dependencies:
Ensure you have a requirements.txt file in the root of your project. If not, create one with the following content:

streamlit
pandas
numpy
scikit-learn
joblib
plotly

Then install:

pip install -r requirements.txt

Obtain Pre-trained Model Files:

This application relies on a pre-trained model. You will need the following files:

salary_prediction_model.joblib

model_metadata.json

Create a folder named models in the root directory of your project (same level as app.py).

Place both salary_prediction_model.joblib and model_metadata.json inside this models folder.

Your project structure should look like this:

Employee-salary-prediction-using-machine-learning-algorithm/
├── app.py
├── requirements.txt
├── models/
│   ├── salary_prediction_model.joblib
│   └── model_metadata.json
└── .gitignore
└── ... (other files)

🚀 Usage
Run the Streamlit application:

streamlit run app.py

Your browser will automatically open the application (usually at http://localhost:8501).

Predict Salary:

Navigate to the "Predict Salary" tab.

Use the sliders and dropdowns to input details for Age, Years of Experience, Gender, Education Level, and Job Title.

Click "Predict My Salary" to see the estimated income, confidence intervals, and market position.

Dataset Overview & Visualizations:

Navigate to the "Dataset Overview & Visualizations" tab.

Upload your adult 3.csv file to generate interactive graphs showing income distribution, and average income by various features.

📂 Project Structure
app.py: The main Streamlit application script.

requirements.txt: Lists all Python dependencies required for the project.

models/: Directory containing the pre-trained machine learning model (.joblib) and its metadata (.json).

adult 3.csv: (Not included in repo by default, but needed for visualizations) The raw dataset used for training and visualizations.

📞 Contact
For any questions or collaborations, feel free to reach out:

Name: Joe Lonney

GitHub: https://github.com/joelonney/Employee-salary-prediction-using-machine-learning-algorithm/blob/master/README.md

IBM Edunet Internship Project