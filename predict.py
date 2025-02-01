import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load trained model
model = tf.keras.models.load_model('crime_severity_mlp.keras')

# Load historical crime data
df_history = pd.read_csv('raw_crime_data.csv')

# Streamlit UI
def main():
    st.title("Crime Severity Prediction System")
    
    # User inputs
    date_rptd = st.date_input("Date Reported")
    date_occ = st.date_input("Date Occurred")
    vict_age = st.number_input("Victim Age", min_value=0, max_value=120, value=30)
    crm_cd_desc = st.text_input("Crime Description")
    area_name = st.text_input("Area Name")
    vict_sex = st.selectbox("Victim Sex", options=["Male", "Female", "Unknown"])
    time_occ = st.number_input("Time Occurred (HHMM)", min_value=0, max_value=2359, value=1200)
    hour_occ = st.number_input("Hour Occurred", min_value=0, max_value=23, value=12)
    lat = st.number_input("Latitude", value=34.0)
    lon = st.number_input("Longitude", value=-118.0)
    
    if st.button("Predict Crime Severity"):
        # Convert date inputs to ordinal
        date_rptd = pd.to_datetime(date_rptd).toordinal()
        date_occ = pd.to_datetime(date_occ).toordinal()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['Crm Cd Desc', 'AREA NAME', 'Vict Sex']
        sample_data = pd.DataFrame([[date_rptd, date_occ, vict_age, crm_cd_desc, area_name, vict_sex, time_occ, hour_occ, lat, lon]],
                                   columns=['Date Rptd', 'DATE OCC', 'Vict Age', 'Crm Cd Desc', 'AREA NAME', 'Vict Sex', 'TIME OCC', 'HOUR OCC', 'LAT', 'LON'])
        
        for col in categorical_cols:
            le = LabelEncoder()
            sample_data[col] = le.fit_transform(sample_data[col])
            label_encoders[col] = le
        
        # Normalize numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_data)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Display prediction
        severity_label = "High Severity" if predicted_class == 1 else "Low Severity"
        st.success(f"Predicted Crime Severity: {severity_label}")

        # Visualization
        st.subheader("Crime Severity Prediction Confidence")
        fig, ax = plt.subplots()
        sns.barplot(x=["Low Severity", "High Severity"], y=predictions[0], ax=ax)
        ax.set_ylabel("Confidence Score")
        st.pyplot(fig)
        
    # Historical crime trends
    st.subheader("Historical Crime Trends")
    df_area = df_history[df_history['AREA NAME'] == area_name]
    if not df_area.empty:
        fig, ax = plt.subplots()
        sns.lineplot(x=df_area['DATE OCC'], y=df_area['Part 1-2'], ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Crime Severity")
        ax.set_title(f"Crime Trends in {area_name}")
        st.pyplot(fig)
    else:
        st.warning("No historical data available for the selected area.")
    
    # Crime density heatmap
    st.subheader("Crime Density Heatmap")
    if not df_history.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(x=df_history['LON'], y=df_history['LAT'], cmap='Reds', fill=True, ax=ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Crime Density Heatmap")
        st.pyplot(fig)
    else:
        st.warning("No crime density data available.")

if __name__ == "__main__":
    main()