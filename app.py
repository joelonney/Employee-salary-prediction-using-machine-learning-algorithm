import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set page configuration for better aesthetics
st.set_page_config(
    page_title="Employee Income Predictor (Adult Dataset)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global variables for encoders (to be used for batch prediction) ---
# Initialize as None, will be set after successful data loading/preprocessing
global_preprocessor = None
global_feature_names = None

# --- 1. Load and Preprocess Dataset Function ---
@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_and_preprocess_data(uploaded_file_buffer):
    """
    Loads the adult.csv dataset and performs preprocessing based on EMPLOYEE.ipynb:
    - Handles missing values ('?').
    - Removes specific workclass categories.
    - Filters outliers for 'age' and 'educational-num'.
    - Transforms 'income' into a numerical target for regression.
    - Drops 'fnlwgt'.
    - Applies OneHotEncoding to categorical features and StandardScaler to numerical features.
    """
    # Ensure the file pointer is at the beginning before reading
    uploaded_file_buffer.seek(0)
    df = pd.read_csv(uploaded_file_buffer)

    # --- Preprocessing based on EMPLOYEE.ipynb ---

    # 1. Handle '?' values by replacing with 'Others' for specific columns
    df['workclass'] = df['workclass'].replace(' ?', 'Others')
    df['occupation'] = df['occupation'].replace(' ?', 'Others')
    df['native-country'] = df['native-country'].replace(' ?', 'Others')

    # Impute any remaining missing categorical values with the mode
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 2. Remove specific 'workclass' categories
    df = df[df['workclass'] != 'Without-pay']
    df = df[df['workclass'] != 'Never-worked']

    # 3. Outlier removal for 'age' and 'educational-num'
    df = df[(df['age'] <= 75) & (df['age'] >= 17)]
    df = df[(df['educational-num'] <= 16) & (df['educational-num'] >= 5)]

    # --- Target Transformation for Regression ---
    # Convert 'income' column to numerical values for regression
    # Assuming <=50K maps to 40000 and >50K maps to 80000
    df['income_numerical'] = df['income'].apply(lambda x: 80000 if x.strip() == '>50K' else 40000)

    # Drop the original 'income' column and 'fnlwgt'
    df = df.drop(columns=['income', 'fnlwgt'])

    # --- Define features for preprocessing ---
    numerical_features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'native-country'
    ]

    # Create a preprocessor pipeline for OneHotEncoding and StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # Fit the preprocessor to the data to get feature names after transformation
    X_for_fit = df.drop(columns=['income_numerical'])
    preprocessor.fit(X_for_fit)
    
    # Get the names of the columns after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_feature_names = numerical_features + list(ohe_feature_names)

    # Transform the data
    X_processed = preprocessor.transform(X_for_fit)
    
    # Convert sparse matrix to dense array
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    df_processed = pd.DataFrame(X_processed, columns=final_feature_names)
    df_processed['income_numerical'] = df['income_numerical'].values # Add target back

    # Return the preprocessor and feature names along with the processed DataFrame
    return df_processed, 'income_numerical', preprocessor, final_feature_names

# --- 2. Train Model Function ---
@st.cache_resource(show_spinner="Training machine learning model...")
def train_model(data, target_column):
    """Trains the Random Forest Regressor model on the preprocessed data."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # The data is already preprocessed by load_and_preprocess_data, so we just need the regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and return metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, mae, rmse, r2

# --- Main Streamlit App ---

st.title("💰 Employee Income Predictor (Adult Dataset)")
st.markdown("""
    Welcome to the Employee Income Prediction Web App!
    This tool uses a Machine Learning model (Random Forest Regressor) to predict an individual's
    approximate income based on various demographic and employment attributes from the **Adult Income Dataset**.

    **Note on Income:** The original dataset categorizes income as `<=50K` or `>50K`. For this regression model,
    we have transformed these categories into numerical values: `$40,000` for `<=50K` and `$80,000` for `>50K`.
    This allows us to predict a continuous "salary" value, which is an approximation for demonstration purposes.
""")

# --- File Uploader for adult 3.csv ---
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload 'adult 3.csv' for training and preprocessing", type="csv")

df_processed = None
target_col_name = None
model = None
mae, rmse, r2 = None, None, None
original_df = None # Initialize original_df here

if uploaded_file is not None:
    # Removed: global global_preprocessor
    # Removed: global global_feature_names
    
    try:
        # Before processing, ensure the file pointer is at the beginning
        uploaded_file.seek(0)
        df_processed, target_col_name, preprocessor_obj, feature_names_list = load_and_preprocess_data(uploaded_file)
        
        # Assign to global variables after successful loading
        # These assignments will update the module-level global variables
        global_preprocessor = preprocessor_obj
        global_feature_names = feature_names_list

        model, mae, rmse, r2 = train_model(df_processed, target_col_name)
        st.success("Data loaded, preprocessed, and model trained successfully!")

        # --- Option to download the preprocessed data ---
        st.subheader("Download Preprocessed Data (Label Encoded CSV)")
        csv_buffer = io.StringIO()
        df_processed.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download cleaned_adult.csv",
            data=csv_buffer.getvalue(),
            file_name="cleaned_adult.csv",
            mime="text/csv",
            help="This CSV contains your data with categorical features one-hot encoded and numerical features scaled."
        )
        st.info("You can download this 'cleaned_adult.csv' file and use it for batch prediction if desired, as per your internship instructions.")

        # --- Prepare original_df for sidebar inputs and visualizations ---
        uploaded_file.seek(0) # Rewind again for original_df
        original_df = pd.read_csv(uploaded_file)
        original_df['workclass'] = original_df['workclass'].replace(' ?', 'Others')
        original_df['occupation'] = original_df['occupation'].replace(' ?', 'Others')
        original_df['native-country'] = original_df['native-country'].replace(' ?', 'Others')
        original_df = original_df[original_df['workclass'] != 'Without-pay']
        original_df = original_df[original_df['workclass'] != 'Never-worked']
        original_df = original_df[(original_df['age'] <= 75) & (original_df['age'] >= 17)]
        original_df = original_df[(original_df['educational-num'] <= 16) & (original_df['educational-num'] >= 5)]
        original_df['income_numerical'] = original_df['income'].apply(lambda x: 80000 if x.strip() == '>50K' else 40000)


    except Exception as e:
        st.error(f"An error occurred during data loading/preprocessing or model training: {e}")
        st.exception(e) # Display full exception for debugging
else:
    st.info("Please upload your 'adult 3.csv' file to get started.")

# Only show prediction and visualization sections if model is trained AND original_df is ready
if model is not None and df_processed is not None and original_df is not None:
    st.sidebar.header("Predict New Individual's Income")

    # --- User Inputs for Single Prediction ---
    unique_workclass = original_df['workclass'].unique()
    unique_education = original_df['education'].unique()
    unique_marital_status = original_df['marital-status'].unique()
    unique_occupation = original_df['occupation'].unique()
    unique_relationship = original_df['relationship'].unique()
    unique_race = original_df['race'].unique()
    unique_gender = original_df['gender'].unique()
    unique_native_country = original_df['native-country'].unique()

    with st.sidebar.form("single_prediction_form"):
        st.subheader("Enter Individual Details")
        input_age = st.slider("Age", int(original_df['age'].min()), int(original_df['age'].max()), 30)
        input_workclass = st.selectbox("Workclass", unique_workclass)
        input_education = st.selectbox("Education", unique_education)
        input_educational_num = st.slider("Educational Num (Years of Education)", int(original_df['educational-num'].min()), int(original_df['educational-num'].max()), 10)
        input_marital_status = st.selectbox("Marital Status", unique_marital_status)
        input_occupation = st.selectbox("Occupation", unique_occupation)
        input_relationship = st.selectbox("Relationship", unique_relationship)
        input_race = st.selectbox("Race", unique_race)
        input_gender = st.selectbox("Gender", unique_gender)
        input_capital_gain = st.number_input("Capital Gain", min_value=0, max_value=int(original_df['capital-gain'].max()), value=0, step=1000)
        input_capital_loss = st.number_input("Capital Loss", min_value=0, max_value=int(original_df['capital-loss'].max()), value=0, step=1000)
        input_hours_per_week = st.slider("Hours per Week", int(original_df['hours-per-week'].min()), int(original_df['hours-per-week'].max()), 40)
        input_native_country = st.selectbox("Native Country", unique_native_country)

        predict_button = st.form_submit_button("Predict Income")

    # --- Single Prediction Logic ---
    if predict_button:
        # Create a DataFrame for the new input, matching the original columns
        new_individual_data = pd.DataFrame([{
            'age': input_age,
            'workclass': input_workclass,
            'education': input_education,
            'educational-num': input_educational_num,
            'marital-status': input_marital_status,
            'occupation': input_occupation,
            'relationship': input_relationship,
            'race': input_race,
            'gender': input_gender,
            'capital-gain': input_capital_gain,
            'capital-loss': input_capital_loss,
            'hours-per-week': input_hours_per_week,
            'native-country': input_native_country
        }])

        try:
            # Use the globally stored preprocessor to transform the new input
            # Check if global_preprocessor is not None before transforming
            if global_preprocessor is not None:
                transformed_input = global_preprocessor.transform(new_individual_data)
                predicted_income = model.predict(transformed_input)[0]
                st.sidebar.success(f"**Predicted Income:** ${predicted_income:,.2f}")
            else:
                st.sidebar.error("Model preprocessor not initialized. Please upload the dataset first.")
        except Exception as e:
            st.sidebar.error(f"Error predicting income for single input: {e}")
            st.sidebar.exception(e)

    st.sidebar.markdown("---")
    st.sidebar.header("Model Performance")
    st.sidebar.info(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
    st.sidebar.info(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    st.sidebar.info(f"**R-squared (R2):** {r2:.4f}")
    st.sidebar.caption("Higher R-squared (closer to 1) indicates a better fit.")

    st.markdown("---")
    st.header("Batch Prediction (Using Pre-encoded CSV)")
    st.markdown("""
        If you have a CSV file that is already preprocessed (like the `cleaned_adult.csv` you can download above),
        you can upload it here to get batch predictions.
        **Important:** This file should contain the numerical/encoded features expected by the model.
    """)
    batch_upload_file = st.file_uploader("Upload pre-encoded CSV for batch prediction", type="csv", key="batch_uploader")

    if batch_upload_file is not None:
        try:
            batch_data = pd.read_csv(batch_upload_file)
            st.write("Uploaded batch data preview:")
            st.dataframe(batch_data.head())

            # Ensure the columns in the uploaded batch data match the training data
            # We need to drop the target column if it exists in the batch data
            if target_col_name in batch_data.columns:
                X_batch = batch_data.drop(columns=[target_col_name])
            else:
                X_batch = batch_data

            # Check if global_feature_names is not None before comparing
            if global_feature_names is not None and not np.array_equal(X_batch.columns, global_feature_names):
                st.warning("Column mismatch detected! The uploaded batch CSV columns do not exactly match the features the model was trained on after preprocessing. Please ensure your batch CSV is correctly pre-encoded.")
                st.info(f"Expected columns: {list(global_feature_names)}")
                st.info(f"Uploaded columns: {list(X_batch.columns)}")
            elif global_preprocessor is None:
                 st.warning("Model preprocessor not initialized. Cannot perform batch prediction.")
            else:
                batch_predictions = model.predict(X_batch)
                batch_data['Predicted_Income'] = batch_predictions.round(2)
                st.subheader("Batch Predictions")
                st.dataframe(batch_data.head())

                csv_output = io.StringIO()
                batch_data.to_csv(csv_output, index=False)
                st.download_button(
                    label="Download Batch Predictions CSV",
                    data=csv_output.getvalue(),
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
                st.success("Batch predictions generated and available for download!")
        except Exception as e:
            st.error(f"An error occurred during batch prediction: {e}")
            st.exception(e)

    # --- Data Exploration and Visualizations (only if data is loaded) ---
    st.header("Dataset Overview and Visualizations")

    st.subheader("First 5 Rows of Preprocessed Data")
    st.dataframe(df_processed.head())

    st.subheader("Dataset Statistics")
    st.dataframe(df_processed.describe())

    st.subheader("Income Distribution (Transformed)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_processed[target_col_name], kde=True, ax=ax, color='lightcoral')
    ax.set_title('Distribution of Transformed Income')
    ax.set_xlabel('Income ($)')
    ax.set_ylabel('Number of Individuals')
    st.pyplot(fig)

    st.subheader("Average Income by Key Features (from Original Data)")
    # Use original_df for visualizations that make sense with original categories
    # This avoids plotting encoded values directly which might not be interpretable
    st.markdown("*(These visualizations use the original categorical values for better readability)*")

    # Average Income by Education Level
    fig_edu, ax_edu = plt.subplots(figsize=(12, 7))
    sns.barplot(x='education', y='income_numerical', data=original_df.groupby('education')['income_numerical'].mean().reset_index().sort_values('income_numerical', ascending=False), ax=ax_edu, palette='viridis')
    ax_edu.set_title('Average Income by Education Level')
    ax_edu.set_xlabel('Education Level')
    ax_edu.set_ylabel('Average Income ($)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_edu)

    # Average Income by Occupation
    fig_occ, ax_occ = plt.subplots(figsize=(14, 8))
    sns.barplot(x='occupation', y='income_numerical', data=original_df.groupby('occupation')['income_numerical'].mean().reset_index().sort_values('income_numerical', ascending=False), ax=ax_occ, palette='magma')
    ax_occ.set_title('Average Income by Occupation')
    ax_occ.set_xlabel('Occupation')
    ax_occ.set_ylabel('Average Income ($)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_occ)

    # Average Income by Gender
    fig_gender, ax_gender = plt.subplots(figsize=(8, 5))
    sns.barplot(x='gender', y='income_numerical', data=original_df.groupby('gender')['income_numerical'].mean().reset_index().sort_values('income_numerical', ascending=False), ax=ax_gender, palette='cividis')
    ax_gender.set_title('Average Income by Gender')
    ax_gender.set_xlabel('Gender')
    ax_gender.set_ylabel('Average Income ($)')
    st.pyplot(fig_gender)

st.markdown("---")
st.caption("Developed for IBM Edunet Internship. This app uses the Adult Income Dataset with a transformed income target for regression demonstration.")
