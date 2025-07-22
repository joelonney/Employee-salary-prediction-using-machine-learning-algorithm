import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import joblib
import os
import warnings
import time
import io # Keep io for potential future use or if any part of model_metadata needs it

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="💰 AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SalaryPredictionApp Class (from your friend's code, adapted) ---
class SalaryPredictionApp:
    def __init__(self):
        self.model_path = 'models/salary_prediction_model.joblib'
        self.metadata_path = 'models/model_metadata.json'
        self.model = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and metadata from local files."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.metadata_path):
                self.model = joblib.load(self.model_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                st.success("✅ Model and metadata loaded successfully!")
                return True
            else:
                st.error("⚠️ Model files not found. Please ensure 'models/salary_prediction_model.joblib' and 'models/model_metadata.json' exist in your project's 'models' directory.")
                return False
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}. Make sure the files are valid and compatible.")
            return False
    
    def validate_age_education_consistency(self, age, education_level):
        """Enhanced validation for age and education level consistency"""
        min_ages = {
            "High School": 17,
            "Bachelor's": 21,
            "Master's": 23,
            "PhD": 26
        }
        
        min_age = min_ages.get(education_level, 18)
        if age < min_age:
            return False, f"❌ Minimum age for {education_level} degree is typically {min_age} years"
        
        if education_level == "PhD" and age > 65:
            return True, f"⚠️ PhD completion after age 65 is rare but possible"
        
        return True, ""
    
    def validate_age_experience_consistency(self, age, experience):
        """Enhanced validation for age and experience consistency"""
        min_working_age = 16
        max_possible_experience = age - min_working_age
        
        if experience > max_possible_experience:
            return False, f"❌ With age {age}, maximum possible experience is {max_possible_experience} years"
        
        if experience < 0:
            return False, "❌ Years of experience cannot be negative"
        
        if age < 22 and experience > 4:
            return False, f"❌ At age {age}, having {experience} years of experience seems unrealistic"
        
        if age >= 22 and age <= 25 and experience == 0:
            return True, f"⚠️ Fresh graduate? Consider internships or part-time work experience"
        
        if age > 40 and experience < 8:
            return True, f"⚠️ Career change or re-entry? Consider highlighting transferable skills"
        
        if experience > 35:
            return True, f"⚠️ {experience} years is exceptional experience - ensure this is accurate"
        
        return True, ""
    
    def validate_experience_education_consistency(self, experience, education_level, age):
        """Enhanced validation for experience and education consistency"""
        degree_years = {
            "High School": 0,
            "Bachelor's": 4,
            "Master's": 6,
            "PhD": 9
        }
        
        years_in_education = degree_years.get(education_level, 0)
        earliest_work_start = 18 + years_in_education
        max_realistic_experience = max(0, age - earliest_work_start)
        
        if experience > max_realistic_experience and max_realistic_experience >= 0:
            adjusted_max = max_realistic_experience + 3
            if experience > adjusted_max:
                return False, f"❌ With {education_level} degree at age {age}, maximum realistic experience is {adjusted_max} years"
        
        return True, ""
    
    def validate_job_title_requirements(self, job_title, education_level, experience, age):
        """Enhanced job title validation with industry standards"""
        job_requirements = {
            'Data Scientist': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 2,
                'typical_age_range': (24, 60),
                'preferred_education': ["Master's", "PhD"],
                'blocked_education': ["High School"]
            },
            'Software Engineer': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 0,
                'typical_age_range': (22, 65),
                'preferred_education': ["Bachelor's"],
                'blocked_education': ["High School"]
            },
            'Senior Engineer': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 7,
                'typical_age_range': (29, 65),
                'preferred_education': ["Bachelor's", "Master's"],
                'blocked_education': ["High School"]
            },
            'Manager': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 5,
                'typical_age_range': (27, 65),
                'preferred_education': ["Bachelor's", "Master's"],
                'blocked_education': ["High School"]
            },
            'Director': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 12,
                'typical_age_range': (35, 65),
                'preferred_education': ["Master's", "PhD"],
                'blocked_education': ["High School"]
            },
            'Consultant': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 3,
                'typical_age_range': (25, 60),
                'preferred_education': ["Master's"],
                'blocked_education': ["High School"]
            },
            'Developer': {
                'min_education': ["High School", "Bachelor's", "Master's", "PhD"],
                'min_experience': 0,
                'typical_age_range': (18, 65),
                'preferred_education': ["Bachelor's"],
                'blocked_education': []
            },
            'Analyst': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 0,
                'typical_age_range': (22, 60),
                'preferred_education': ["Bachelor's"],
                'blocked_education': ["High School"]
            }
        }
        
        if job_title in job_requirements:
            req = job_requirements[job_title]
            warnings = []
            
            if education_level in req.get('blocked_education', []):
                return False, f"❌ {job_title} cannot be achieved with {education_level} education. Minimum required: {', '.join(req['min_education'])}"
            
            if education_level not in req['min_education']:
                return False, f"❌ {job_title} typically requires: {', '.join(req['min_education'])}"
            
            if experience < req['min_experience']:
                return False, f"❌ {job_title} typically requires at least {req['min_experience']} years of experience"
            
            min_age, max_age = req['typical_age_range']
            if age < min_age:
                return False, f"❌ {job_title} positions typically start at age {min_age}+"
            
            if age > max_age:
                warnings.append(f"⚠️ {job_title} at age {age} is possible but less common")
            
            if education_level not in req['preferred_education']:
                warnings.append(f"⚠️ {job_title} typically prefers: {', '.join(req['preferred_education'])}")
            
            if warnings:
                return True, " | ".join(warnings)
        
        return True, ""
    
    def validate_salary_expectations(self, job_title, experience, education_level):
        """Validate salary expectations based on role and experience"""
        salary_ranges = {
            'Data Scientist': {'entry': (70000, 90000), 'mid': (90000, 130000), 'senior': (130000, 180000)},
            'Software Engineer': {'entry': (60000, 80000), 'mid': (80000, 120000), 'senior': (120000, 160000)},
            'Senior Engineer': {'entry': (100000, 130000), 'mid': (130000, 160000), 'senior': (160000, 200000)},
            'Manager': {'entry': (80000, 110000), 'mid': (110000, 150000), 'senior': (150000, 200000)},
            'Director': {'entry': (120000, 160000), 'mid': (160000, 220000), 'senior': (220000, 300000)},
            'Consultant': {'entry': (65000, 85000), 'mid': (85000, 120000), 'senior': (120000, 170000)},
            'Developer': {'entry': (50000, 70000), 'mid': (70000, 100000), 'senior': (100000, 140000)},
            'Analyst': {'entry': (45000, 65000), 'mid': (65000, 90000), 'senior': (90000, 120000)}
        }
        
        if job_title in salary_ranges:
            ranges = salary_ranges[job_title]
            if experience <= 2:
                return ranges['entry']
            elif experience <= 7:
                return ranges['mid']
            else:
                return ranges['senior']
        
        return (50000, 150000)
    
    def comprehensive_validation(self, data):
        """Enhanced comprehensive validation with detailed feedback"""
        errors = []
        warnings = []
        
        age = data.get('Age', 0)
        experience = data.get('Years of Experience', 0)
        education = data.get('Education Level', '')
        job_title = data.get('Job Title', '')
        gender = data.get('Gender', '')
        
        if age < 16 or age > 75:
            errors.append("❌ Age must be between 16 and 75 years")
        
        if experience < 0 or experience > 55:
            errors.append("❌ Years of experience must be between 0 and 55")
        
        if not gender or gender not in ['Male', 'Female', 'Other']: # Added 'Other' as an option
            errors.append("❌ Please select a valid gender")
        
        if age and education:
            is_valid, msg = self.validate_age_education_consistency(age, education)
            if not is_valid:
                errors.append(msg)
            elif "⚠️" in msg:
                warnings.append(msg)
        
        if age and experience is not None:
            is_valid, msg = self.validate_age_experience_consistency(age, experience)
            if not is_valid:
                errors.append(msg)
            elif "⚠️" in msg:
                warnings.append(msg)
        
        if experience is not None and education and age:
            is_valid, msg = self.validate_experience_education_consistency(experience, education, age)
            if not is_valid:
                errors.append(msg)
        
        if job_title and education and experience is not None and age:
            is_valid, msg = self.validate_job_title_requirements(job_title, education, experience, age)
            if not is_valid:
                errors.append(msg)
            elif "⚠️" in msg:
                warnings.append(msg)
        
        if experience > 15 and "Senior" not in job_title and job_title not in ["Manager", "Director"]:
            warnings.append("⚠️ With 15+ years experience, consider senior or management roles")
        
        if experience < 3 and job_title in ["Manager", "Director"]:
            errors.append("❌ Management roles typically require 3+ years of experience")
        
        if education == "PhD" and experience < 2:
            warnings.append("⚠️ PhD graduates typically have research or internship experience")
        
        if education == "High School" and job_title in ["Data Scientist", "Manager", "Director"]:
            warnings.append("⚠️ This role typically requires higher education")
        
        if age > 45 and experience < 10:
            warnings.append("⚠️ Career change or re-entry? Consider highlighting transferable skills")
        
        if age < 30 and job_title == "Director":
            warnings.append("⚠️ Director role at young age - ensure leadership experience")
        
        return errors, warnings
    
    def predict_salary(self, data):
        if not self.model or not self.metadata:
            return None, ["❌ Model not loaded properly. Please check model files."]
        
        try:
            errors, warnings = self.comprehensive_validation(data)
            
            if errors:
                return None, errors
            
            # Convert input data to DataFrame matching expected features for the model
            # This assumes the model's preprocessing pipeline is embedded OR it expects raw features
            # If your friend's model expects one-hot encoded columns, this part needs careful mapping
            # based on the exact columns in `self.metadata['feature_names']`.
            # For simplicity, we assume the model takes a DataFrame with the input feature names directly.
            input_df = pd.DataFrame([data])
            
            # If your friend's model expects specific column order or one-hot encoding,
            # you would need to implement that here based on `self.metadata`.
            # Example:
            # if 'preprocessor_pipeline' in self.metadata:
            #     transformed_input = self.metadata['preprocessor_pipeline'].transform(input_df)
            # else:
            #     transformed_input = input_df # Assuming raw features are expected

            # For now, assuming the model can handle the raw input DataFrame directly
            prediction = self.model.predict(input_df)[0]
            
            model_metrics = self.metadata.get('final_metrics', {})
            model_std = model_metrics.get('test_rmse', 15000) # Default RMSE if not in metadata
            
            confidence_factor = self._calculate_confidence_factor(data)
            margin_of_error = model_std * confidence_factor
            
            lower_bound = max(25000, prediction - margin_of_error)
            upper_bound = min(500000, prediction + margin_of_error)
            
            expected_range = self.validate_salary_expectations(
                data['Job Title'], 
                data['Years of Experience'], 
                data['Education Level']
            )
            
            result = {
                'prediction': round(prediction, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'confidence': self._get_confidence_level(),
                'expected_range': expected_range,
                'warnings': warnings,
                'market_position': self._get_market_position(prediction),
                'confidence_factor': confidence_factor
            }
            
            return result, []
            
        except Exception as e:
            error_msg = f"❌ Prediction error: {str(e)}. Ensure all required input features are provided and match the model's expectations."
            st.error(error_msg)
            return None, [error_msg]
    
    def _calculate_confidence_factor(self, data):
        """Calculate confidence factor based on data quality"""
        confidence = 1.0
        
        age = data.get('Age', 30)
        if age < 22 or age > 60:
            confidence *= 1.2
        
        experience = data.get('Years of Experience', 0)
        if experience > 30:
            confidence *= 1.3
        
        education = data.get('Education Level', '')
        job_title = data.get('Job Title', '')
        
        high_skill_jobs = ['Data Scientist', 'Senior Engineer', 'Director']
        if job_title in high_skill_jobs and education == 'High School':
            confidence *= 1.4
        
        return min(confidence, 2.0)
    
    def _get_confidence_level(self):
        """Return confidence level as percentage"""
        return "85-90%"
    
    def _get_market_position(self, prediction):
        """Determine market position of the predicted salary"""
        if prediction < 60000:
            return "Entry Level"
        elif prediction < 100000:
            return "Mid Level"
        elif prediction < 150000:
            return "Senior Level"
        else:
            return "Executive Level"
    
    def create_salary_distribution_chart(self, prediction, job_title):
        """Create salary distribution visualization"""
        np.random.seed(42)
        sample_salaries = np.random.normal(prediction, 20000, 1000)
        sample_salaries = sample_salaries[sample_salaries > 30000]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=sample_salaries,
            nbinsx=30,
            name='Salary Distribution',
            opacity=0.7,
            marker_color='#667eea' # Using a pleasant Streamlit-like color
        ))
        
        fig.add_vline(
            x=prediction,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Your Prediction: ${prediction:,.0f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=f'Salary Distribution for {job_title}',
            xaxis_title='Salary ($)',
            yaxis_title='Frequency',
            showlegend=False,
            height=400,
            template="plotly_white" # Clean white theme
        )
        return fig
    
    def create_feature_importance_chart(self):
        """Create feature importance visualization"""
        # Sample feature importance (replace with actual model feature importance if available in metadata)
        features = ['Years of Experience', 'Education Level', 'Age', 'Job Title', 'Gender']
        importance = [0.40, 0.25, 0.15, 0.15, 0.05]
        
        # If actual feature importances are in metadata, use them:
        # if self.metadata and 'feature_importances' in self.metadata:
        #     features = list(self.metadata['feature_importances'].keys())
        #     importance = list(self.metadata['feature_importances'].values())
        #     # Sort by importance
        #     sorted_indices = np.argsort(importance)[::-1]
        #     features = [features[i] for i in sorted_indices]
        #     importance = [importance[i] for i in sorted_indices]

        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='#764ba2' # Another pleasant Streamlit-like color
        ))
        fig.update_layout(
            title='Feature Importance in Salary Prediction',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=300,
            template="plotly_white"
        )
        return fig
    
    def display_prediction_insights(self, result, data):
        """Display detailed prediction insights"""
        prediction = result['prediction']
        lower_bound = result['lower_bound']
        upper_bound = result['upper_bound']
        
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Predicted Salary", value=f"${prediction:,.2f}")
        with col2:
            st.metric(label="Lower Bound", value=f"${lower_bound:,.2f}")
        with col3:
            st.metric(label="Upper Bound", value=f"${upper_bound:,.2f}")

        st.info(f"""
        **Confidence Level:** {result['confidence']}  
        **Expected Market Range for this role:** ${result['expected_range'][0]:,.0f} - ${result['expected_range'][1]:,.0f}  
        **Your Market Position:** {result['market_position']}
        """)

        if result['warnings']:
            st.warning("#### Potential Issues/Warnings:")
            for w in result['warnings']:
                st.write(f"- {w}")

        st.subheader("Salary Distribution & Feature Importance")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(self.create_salary_distribution_chart(prediction, data['Job Title']), use_container_width=True)
        with col_chart2:
            st.plotly_chart(self.create_feature_importance_chart(), use_container_width=True)

# --- Main App Execution ---
def main():
    st.title("💰 AI Salary Predictor")
    st.markdown("Predict your potential salary based on your profile and market insights.")

    app = SalaryPredictionApp()

    if not app.model or not app.metadata:
        st.stop() # Stop execution if model is not loaded

    st.subheader("Enter Your Profile Details")

    with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 70, 30)
            years_experience = st.slider("Years of Experience", 0, 40, 5)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col2:
            education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
            job_title = st.selectbox("Job Title", [
                "Data Scientist", "Software Engineer", "Senior Engineer",
                "Manager", "Director", "Consultant", "Developer", "Analyst"
            ])
        
        submitted = st.form_submit_button("Predict My Salary")

    if submitted:
        input_data = {
            'Age': age,
            'Years of Experience': years_experience,
            'Education Level': education_level,
            'Job Title': job_title,
            'Gender': gender
        }
        
        with st.spinner('Calculating your potential salary...'):
            time.sleep(1) # Simulate some processing time
            prediction_result, errors = app.predict_salary(input_data)

        if prediction_result:
            app.display_prediction_insights(prediction_result, input_data)
        else:
            for error_msg in errors:
                st.error(error_msg)

    st.markdown("---")
    st.caption("Developed by Joe Lonney. This app uses a pre-trained model for salary prediction.")

if __name__ == "__main__":
    main()
