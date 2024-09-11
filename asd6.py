import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from io import BytesIO

# Path to save the model and data frame file
model_folder = 'C:\\asd6'
model_path = os.path.join(model_folder, 'ASD6.pkl')
data_frame_template_path = 'The data frame file to be analyzed.xlsx'

# Function to train the model and save it as "ASD6"
def train_and_save_model():
    try:
        # Ensure the folder exists
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        # Load the training data from the specified path
        file_path = r'C:\asd6\final_classified_loss_with_reasons_60_percent_ordered.xlsx'
        data = pd.read_excel(file_path)
        
        # Prepare the features and target
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        y = data['Loss_Status'].apply(lambda x: 1 if x == 'Loss' else 0)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the RandomForestClassifier model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Save the trained model to the asd6 folder
        joblib.dump(model, model_path)
        st.success(f"Model trained and saved as {model_path}")
    
    except Exception as e:
        st.error(f"An error occurred while training the model: {str(e)}")

# Train the model automatically when the system starts if not already trained
if not os.path.exists(model_path):
    train_and_save_model()

# Function to add reasons based on analysis conditions
def add_loss_reason(row):
    if row['V1'] == 0 and row['A1'] > 0:
        return 'Loss due to zero voltage with current on V1'
    elif row['V2'] == 0 and row['A2'] > 0:
        return 'Loss due to zero voltage with current on V2'
    elif row['V3'] == 0 and row['A3'] > 0:
        return 'Loss due to zero voltage with current on V3'
    elif row['V1'] == 0 and row['A1'] == 0 and abs(row['A2'] - row['A3']) > 0.6 * max(row['A2'], row['A3']):
        return 'Loss due to zero voltage and current on V1 with imbalance between A2 and A3'
    elif row['V2'] == 0 and row['A2'] == 0 and abs(row['A1'] - row['A3']) > 0.6 * max(row['A1'], row['A3']):
        return 'Loss due to zero voltage and current on V2 with imbalance between A1 and A3'
    elif row['V3'] == 0 and row['A3'] == 0 and abs(row['A1'] - row['A2']) > 0.6 * max(row['A1'], row['A2']):
        return 'Loss due to zero voltage and current on V3 with imbalance between A1 and A2'
    elif row['V1'] < 10 and row['A1'] > 0:
        return 'Loss due to low voltage with current on V1'
    elif row['V2'] < 10 and row['A2'] > 0:
        return 'Loss due to low voltage with current on V2'
    elif row['V3'] < 10 and row['A3'] > 0:
        return 'Loss due to low voltage with current on V3'
    elif abs(row['A1'] - row['A2']) > 0.6 * max(row['A1'], row['A2']) and row['A3'] == 0:
        return 'Loss due to high current difference between A1 and A2 with zero A3'
    else:
        return 'No specific loss condition met'

# Function to analyze data and separate high-priority loss cases
def analyze_data(data):
    try:
        # Check if 'Meter Number' column exists, otherwise raise an error
        if 'Meter Number' not in data.columns:
            st.error("The file does not contain the 'Meter Number' column. Please check your file.")
            return
        
        # Load the trained model
        model = joblib.load(model_path)
        
        # Ensure the correct order of columns
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        
        # Predict using the trained model
        predictions = model.predict(X)
        
        # Add the predictions to the dataframe
        data['Predicted_Loss'] = predictions
        
        # Filter only the rows where loss is detected
        loss_data = data[data['Predicted_Loss'] == 1].copy()  # Ensure we are working with a copy
        
        # Add reason for the prediction based on conditions
        loss_data['Reason'] = loss_data.apply(add_loss_reason, axis=1)
        
        # Select high-priority cases based on reasons
        high_priority_cases = loss_data[loss_data['Reason'].str.contains('zero voltage')]
        
        # Count the number of loss cases
        loss_count = len(loss_data)
        high_priority_count = len(high_priority_cases)
        
        # Display the result in the app
        st.write(f'Number of loss cases detected: {loss_count}')
        st.write(f'Number of high-priority loss cases: {high_priority_count}')
        
        # Display all loss cases
        st.subheader("Detected Loss Cases")
        st.dataframe(loss_data[['Meter Number', 'V1', 'V2', 'V3', 'A1', 'A2', 'A3', 'Predicted_Loss', 'Reason']])
        
        # Display high-priority cases
        st.subheader("High-Priority Loss Cases")
        st.dataframe(high_priority_cases[['Meter Number', 'V1', 'V2', 'V3', 'A1', 'A2', 'A3', 'Predicted_Loss', 'Reason']])
        
        # Export all loss cases to an Excel file
        output_loss = BytesIO()
        with pd.ExcelWriter(output_loss, engine='xlsxwriter') as writer:
            loss_data.to_excel(writer, index=False)
        output_loss.seek(0)
        
        # Export high-priority cases to an Excel file
        output_high_priority = BytesIO()
        with pd.ExcelWriter(output_high_priority, engine='xlsxwriter') as writer:
            high_priority_cases.to_excel(writer, index=False)
        output_high_priority.seek(0)
        
        # Create a download button for all loss cases
        st.download_button(label="Download All Loss Cases",
                           data=output_loss,
                           file_name='all_loss_cases.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
        # Create a download button for high-priority cases
        st.download_button(label="Download High-Priority Loss Cases",
                           data=output_high_priority,
                           file_name='high_priority_loss_cases.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Streamlit app setup
st.title("التنبأ بحالات الفاقد المحتمله")

# Add a button to download the template file
with open(data_frame_template_path, 'rb') as template_file:
    template_data = template_file.read()
st.download_button(label="Download Data Frame Template", data=template_data, file_name='The_data_frame_file_to_be_analyzed.xlsx')

# Upload file to analyze (auto-analyze on upload)
st.header("Analyze Data")
uploaded_analyze_file = st.file_uploader("Upload data to analyze (Excel)", type=["xlsx"], key="analyze")
if uploaded_analyze_file is not None:
    try:
        analyze_data_df = pd.read_excel(uploaded_analyze_file)
        analyze_data(analyze_data_df)  # Automatically trigger analysis after upload
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
