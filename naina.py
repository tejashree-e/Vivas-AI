import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests
from datetime import datetime

# 🔹 Define model path
model_path = "water_prediction_model.pkl"
encoder_path = "label_encoders.pkl"

# 🔹 Replace with your OpenWeatherMap API Key
API_KEY = "b1ca3cb4e003bd87059b5969f3811472"

# 🔹 Default Water Requirements for Major Tamil Nadu Crops (L/m² per day)
default_crop_water = {
    "paddy": 12, "rice": 12, "sugarcane": 15, "banana": 10, "coconut": 8,
    "cotton": 7, "ragi": 4, "finger millet": 4, "groundnut": 5, "tomato": 6,
    "brinjal": 7, "eggplant": 7, "turmeric": 8
}

# 🔹 Default Irrigation Efficiency
default_irrigation_efficiency = {
    "drip": 0.9, "sprinkler": 0.75, "flood": 0.5, "furrow": 0.6, "manual": 0.4, "pump": 0.7
}

# 🔹 Default Soil Retention Factors
default_soil_retention = {"clay": 0.8, "loam": 1.0, "sandy": 1.2}

# 🔹 Welcome Message
print("🌟 கட்சோ விவாஸ்-AI ✨ | Welcome to VIVAS-AI 🌟")

# 🔹 Check if the AI model exists, if not, retrain it
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    print("⚠️ Model or encoders missing! Retraining AI Model...")

    # 🔹 Dummy Training Data (Replace with real dataset later)
    data = {
        "Crop_Type": ["paddy", "wheat", "cotton", "banana", "sugarcane"],
        "Soil_Type": ["clay", "loam", "sandy", "alluvial", "red loam"],
        "Irrigation_Method": ["drip", "flood", "sprinkler", "manual", "pump"],
        "Temperature": [30, 25, 35, 28, 32],
        "Humidity": [80, 65, 50, 90, 85],
        "Rainfall": [10, 5, 0, 15, 7],
        "Water_Needed": [700, 500, 600, 800, 750]
    }
    df = pd.DataFrame(data)

    # 🔹 Encode categorical variables
    label_encoders = {}
    for col in ["Crop_Type", "Soil_Type", "Irrigation_Method"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoders

    # 🔹 Features & Target
    X = df[["Crop_Type", "Soil_Type", "Irrigation_Method", "Temperature", "Humidity", "Rainfall"]]
    y = df["Water_Needed"]

    # 🔹 Train the AI Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 🔹 Save the AI Model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # 🔹 Save Label Encoders
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoders, f)

    print("✅ Model and encoders retrained & saved!")

# 🔹 Load the AI Model & Encoders
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoder_path, "rb") as f:
    label_encoders = pickle.load(f)

print("🎉 AI Model Loaded Successfully!")

# 🔹 Function to Predict Water Requirement
def predict_water(crop, soil, irrigation, temp, humidity, rainfall):
    try:
        # Ensure valid category names
        if crop not in label_encoders["Crop_Type"].classes_:
            crop = label_encoders["Crop_Type"].classes_[0]
        if soil not in label_encoders["Soil_Type"].classes_:
            soil = label_encoders["Soil_Type"].classes_[0]
        if irrigation not in label_encoders["Irrigation_Method"].classes_:
            irrigation = label_encoders["Irrigation_Method"].classes_[0]

        # Encode Inputs
        crop_encoded = label_encoders["Crop_Type"].transform([crop])[0]
        soil_encoded = label_encoders["Soil_Type"].transform([soil])[0]
        irrigation_encoded = label_encoders["Irrigation_Method"].transform([irrigation])[0]

        # Prepare feature array **WITH COLUMN NAMES**
        features = pd.DataFrame([[crop_encoded, soil_encoded, irrigation_encoded, temp, humidity, rainfall]],
                                columns=["Crop_Type", "Soil_Type", "Irrigation_Method", "Temperature", "Humidity", "Rainfall"])

        # Predict water requirement
        predicted_water = model.predict(features)[0]

        return f"🚰 **Predicted Water Need:** {predicted_water:.2f} L/m² per day"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 🔹 Get user inputs
city = input("\n🌍 இடம் (Location): ").strip().lower()
crop = input("\n🌿 பயிர் பெயர் (Crop Name): ").strip().lower()
soil_type = input("\n🌱 மண் வகை (Soil Type - Clay/Loam/Sandy etc.): ").strip().lower()
irrigation_mode = input("\n💧 பாசன முறை (Irrigation Mode - Drip/Sprinkler/Flood etc.): ").strip().lower()

# 🔹 Auto-Estimate Water Efficiency
irrigation_efficiency = default_irrigation_efficiency.get(irrigation_mode, 0.6)

# 🔹 Auto-Estimate Soil Retention
soil_retention_factor = default_soil_retention.get(soil_type, 1.0)

# 🔹 Check crop water requirement
crop_water_need = default_crop_water.get(crop, 6)

# 🔹 Get latitude and longitude for the city
geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
geo_response = requests.get(geo_url).json()

if not geo_response:
    print(f"\n🚨 Error: City '{city}' not found.")
    exit()

lat, lon = geo_response[0]["lat"], geo_response[0]["lon"]

# 🔹 Fetch 5-day weather forecast
forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
forecast_response = requests.get(forecast_url).json()

if forecast_response.get("cod") != "200":
    print(f"\n🚨 Error: {forecast_response.get('message', 'Invalid city name')}")
    exit()

# 🔹 Calculate Total Rainfall for the Next 3 Days
total_rainfall = sum(forecast.get("rain", {}).get("3h", 0) for forecast in forecast_response["list"] if "rain" in forecast)

# 🔹 Adjust for Rainfall
final_water_requirement = max((crop_water_need / irrigation_efficiency) * soil_retention_factor - total_rainfall, 0)

# 🔹 Display Results
print("\n🌱 **பாசன பேர்வுரை (Irrigation Recommendation)** 🌱")
print(f"\n🌍 இடம் (Location): {city.title()} (Lat: {lat}, Lon: {lon})")
print(f"\n💧 பாசன திறன் (Irrigation Efficiency): {irrigation_efficiency * 100:.0f}%")
print(f"\n🌿 மண் நீர் பிடிப்பு காரணி (Soil Retention Factor): {soil_retention_factor}")
print(f"\n🌧️ அடுத்த 3 நாட்களில் மழை (Rainfall in next 3 days): {total_rainfall:.2f} mm")
print(f"\n🚰 நீர் தேவை (Water Needed): {final_water_requirement:.2f} L/m² per day\n")