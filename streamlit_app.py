import streamlit as st
import joblib
import numpy as np


scaler = joblib.load('scaler.pkl')
knn_model = joblib.load('knn_model.pkl')


def preprocess_input(passenger_data):
    scaled_features = scaler.transform([[ 
       passenger_data['ID'],
        passenger_data['Age'],
        passenger_data['Flight Distance'],
        passenger_data['Departure Delay'],
        passenger_data['Arrival Delay'],
        passenger_data['Departure and Arrival Time Convenience'],
        passenger_data['Ease of Online Booking'],
        passenger_data['Check-in Service'],
        passenger_data['Online Boarding'],
        passenger_data['Gate Location'],
        passenger_data['On-board Service'],
        passenger_data['Seat Comfort'],
        passenger_data['Leg Room Service'],
        passenger_data['Cleanliness'],
        passenger_data['Food and Drink'],
        passenger_data['In-flight Service'],
        passenger_data['In-flight Wifi Service'],
        passenger_data['In-flight Entertainment'],
        passenger_data['Baggage Handling']
    ]])
    return scaled_features
st.image("https://media.npr.org/assets/img/2021/10/06/gettyimages-1302813215_wide-a248aa0418c5154e72d6a555f556bf5d99e7cac7.jpg", use_column_width=True)
st.title("Passenger Satisfaction Prediction")

idno = st.number_input("Ticket No", value=0)
gender = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=0, max_value=100, value=30)
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
class_type = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
flight_distance = st.number_input("Flight Distance", value=0)
departure_delay = st.number_input("Departure Delay", value=0)
arrival_delay = st.number_input("Arrival Delay", value=0)
time_convenience = st.slider("Departure and Arrival Time Convenience", min_value=0, max_value=5, value=0)
ease_booking = st.slider("Ease of Online Booking", min_value=0, max_value=5, value=0)
checkin_service = st.slider("Check-in Service", min_value=0, max_value=5, value=0)
online_boarding = st.slider("Online Boarding", min_value=0, max_value=5, value=0)
gate_location = st.slider("Gate Location", min_value=0, max_value=5, value=0)
onboard_service = st.slider("On-board Service", min_value=0, max_value=5, value=0)
seat_comfort = st.slider("Seat Comfort", min_value=0, max_value=5, value=0)
leg_room_service = st.slider("Leg Room Service", min_value=0, max_value=5, value=0)
cleanliness = st.slider("Cleanliness", min_value=0, max_value=5, value=0)
food_drink = st.slider("Food and Drink", min_value=0, max_value=5, value=0)
inflight_service = st.slider("In-flight Service", min_value=0, max_value=5, value=0)
wifi_service = st.slider("In-flight Wifi Service", min_value=0, max_value=5, value=0)
entertainment = st.slider("In-flight Entertainment", min_value=0, max_value=5, value=0)
baggage_handling = st.slider("Baggage Handling", min_value=0, max_value=5, value=0)


if st.button("Predict Satisfaction"):
    
    passenger_data = {
        'ID': idno,
        'Gender': gender,
        'Age': age,
        'Customer Type': customer_type,
        'Type of Travel': type_of_travel,
        'Class': class_type,
        'Flight Distance': flight_distance,
        'Departure Delay': departure_delay,
        'Arrival Delay': arrival_delay,
        'Departure and Arrival Time Convenience': time_convenience,
        'Ease of Online Booking': ease_booking,
        'Check-in Service': checkin_service,
        'Online Boarding': online_boarding,
        'Gate Location': gate_location,
        'On-board Service': onboard_service,
        'Seat Comfort': seat_comfort,
        'Leg Room Service': leg_room_service,
        'Cleanliness': cleanliness,
        'Food and Drink': food_drink,
        'In-flight Service': inflight_service,
        'In-flight Wifi Service': wifi_service,
        'In-flight Entertainment': entertainment,
        'Baggage Handling': baggage_handling
    }
   
    passenger_features = preprocess_input(passenger_data)
  
    satisfaction_prediction = knn_model.predict(passenger_features)
    st.write(satisfaction_prediction)