import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

with st.sidebar:
    st.header("About Me")
    st.write("Hi, I'm Sam! A passionate Data Scientist, Data Analyst, and Python Developer.")
    st.write("📧 wambugumuriukisam@gmail.com")
    st.write("[LinkedIn](https://github.com/samintelguru) | [GitHub](https://github.com/samintelguru)")

# Load dataset (ensure 'Student_Performance_Factors.csv' is in your PyCharm project folder)
try:
    df = pd.read_csv('StudentPerformanceFactors.csv')
except FileNotFoundError:
    st.error("Dataset file not found. Ensure 'Student_Performance_Factors.csv' is in the project folder.")
    st.stop()

# Step 1: Create a copy for EDA to preserve original categorical columns
df_eda = df.copy()

# Step 2: Define columns
numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Physical_Activity', 'Exam_Score']
categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                    'Internet_Access', 'Tutoring', 'Family_Income', 'Teacher_Quality',
                    'School_Type', 'Peer_Influence', 'Gender', 'Distance_from_Home',
                    'Parental_Education_Level']
ordinal_cols = ['Motivation_Level', 'Learning_Disabilities']

# Step 3: Handle missing values
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())
for col in categorical_cols + ordinal_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Step 4: Encode categorical variables
# One-hot encoding for nominal variables
present_categorical_cols = [col for col in categorical_cols if col in df.columns]
if present_categorical_cols:
    df = pd.get_dummies(df, columns=present_categorical_cols, drop_first=True)
else:
    st.warning("No categorical columns found for one-hot encoding.")

# Label encoding for ordinal variables
le_motivation = LabelEncoder()
le_disabilities = LabelEncoder()
if 'Motivation_Level' in df.columns:
    df['Motivation_Level'] = le_motivation.fit_transform(df['Motivation_Level'])
if 'Learning_Disabilities' in df.columns:
    df['Learning_Disabilities'] = le_disabilities.fit_transform(df['Learning_Disabilities'])

# Step 5: Cap outliers for numeric columns
for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# Step 6: Train model
X = df.drop(columns=['Exam_Score'])
y = df['Exam_Score']
try:
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
except ValueError as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# Streamlit app
st.title('Student Exam Score Predictor (SDG 4)')
st.write('Predict exam scores to support quality education')

# Step 7: Inputs for prediction
hours_studied = st.slider('Hours Studied/Week', 0, 50, 20)
attendance = st.slider('Attendance (%)', 0, 100, 80)
sleep_hours = st.slider('Sleep Hours/Night', 0, 12, 7)
motivation = st.selectbox('Motivation Level', ['Low', 'Medium', 'High'])
school_type = st.selectbox('School Type', ['Public', 'Private'])
gender = st.selectbox('Gender', ['Male', 'Female'])
access = st.selectbox('Access to Resources', ['Low', 'Medium', 'High'])
parental_involvement = st.selectbox('Parental Involvement', ['Low', 'Medium', 'High'])
extracurricular = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
internet_access = st.selectbox('Internet Access', ['Yes', 'No'])
parental_education = st.selectbox('Parental Education Level', ['High School', 'College', 'Graduate'])

# Prepare input data
input_data = pd.DataFrame({
    'Hours_Studied': [hours_studied],
    'Attendance': [attendance],
    'Sleep_Hours': [sleep_hours],
    'Motivation_Level': [le_motivation.transform([motivation])[0]],
    'School_Type_Private': [1 if school_type == 'Private' else 0],
    'Gender_Male': [1 if gender == 'Male' else 0],
    'Access_to_Resources_Medium': [1 if access == 'Medium' else 0],
    'Access_to_Resources_High': [1 if access == 'High' else 0],
    'Parental_Involvement_Medium': [1 if parental_involvement == 'Medium' else 0],
    'Parental_Involvement_High': [1 if parental_involvement == 'High' else 0],
    'Extracurricular_Activities_Yes': [1 if extracurricular == 'Yes' else 0],
    'Internet_Access_Yes': [1 if internet_access == 'Yes' else 0],
    'Parental_Education_Level_College': [1 if parental_education == 'College' else 0],
    'Parental_Education_Level_Graduate': [1 if parental_education == 'Graduate' else 0]
})

# Ensure input_data matches X's columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[X.columns]

# Predict
prediction = model.predict(input_data)[0]
st.write(f'Predicted Exam Score: {prediction:.1f}')

# Step 8: EDA Visualizations
st.subheader('Exploratory Data Analysis')
st.write('Key insights into factors affecting student performance')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_eda[['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Exam_Score']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation of Numeric Factors')
st.pyplot(plt)

# Box plot: Parental Involvement (using df_eda)
if 'Parental_Involvement' in df_eda.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Parental_Involvement', y='Exam_Score', data=df_eda)
    plt.title('Exam Score by Parental Involvement (SDG 4)')
    st.pyplot(plt)
else:
    st.warning("Parental_Involvement column not found for box plot.")

# Box plot: Parental Education Level (using df_eda)
if 'Parental_Education_Level' in df_eda.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Parental_Education_Level', y='Exam_Score', data=df_eda)
    plt.title('Exam Score by Parental Education Level (SDG 4)')
    st.pyplot(plt)
else:
    st.warning("Parental_Education_Level column not found for box plot.")

# Histogram: Exam Score
plt.figure(figsize=(8, 5))
sns.histplot(df_eda['Exam_Score'], bins=20, kde=True)
plt.title('Distribution of Exam Scores')
st.pyplot(plt)

# Feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title('Feature Importance for Exam Score')
st.pyplot(plt)

# Feature importance plot (enhanced)
plt.figure(figsize=(8, 5))
# Create DataFrame for sorting
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)  # Top 10 features
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Top 10 Features Influencing Exam Score (SDG 4)')
for i, v in enumerate(importance_df['Importance']):
    plt.text(v, i, f'{v:.3f}', va='center')  # Add importance scores
st.pyplot(plt)

st.write('SDG 4 Insight: High parental involvement, education level, and study hours boost scores, suggesting targeted support programs.')