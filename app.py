from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import ollama

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

translations = {
    "en": {
        "error": "Please enter a message.",
        "switch": "Switch to Tamil",
        "fallback": "I can help with farming! Ask me about weather, market prices, crop advice, or government schemes.",
        "market_prefix": "Market price for"
    },
    "ta": {
        "error": "தயவுசெய்து ஒரு செய்தியை உள்ளிடவும்.",
        "switch": "ஆங்கிலத்திற்கு மாற்று",
        "fallback": "நான் விவசாயத்துக்கு உதவலாம்! வானிலை, சந்தை விலை, பயிர் அறிவுரை அல்லது அரசு திட்டங்களைப் பற்றி என்னிடம் கேளுங்கள்.",
        "market_prefix": "சந்தை விலை"
    }
}

def get_location_info(lat, lon):
    try:
        response = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}")
        data = response.json()
        return data.get("display_name", "Unknown location")
    except Exception:
        return "Unknown location"

def get_weather(lat, lon, lang):
    try:
        response = requests.get(f"https://wttr.in/{lat},{lon}?format=%C+%t&lang={lang}")
        return response.text.strip()
    except Exception:
        return translations[lang]["error"]

import requests
from bs4 import BeautifulSoup


def get_market_price(crop, lang="en"):
    url = "https://agmarknet.gov.in/"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "மார்க்கெட் விலை தகவலை பெற முடியவில்லை." if lang == "ta" else "Error fetching market price data."

    soup = BeautifulSoup(response.text, "html.parser")

    # Locate the table containing market prices (Ensure this matches actual structure)
    table = soup.find("table")  # Adjust if necessary
    if not table:
        return "விலை தரவுகள் கிடைக்கவில்லை." if lang == "ta" else "Market price data not found."

    crop = crop.lower()  # Convert input to lowercase for case-insensitive search

    # Loop through table rows
    for row in table.find_all("tr")[1:]:  # Skip header row
        columns = row.find_all("td")
        if len(columns) > 1 and crop in columns[0].text.lower():
            price = columns[1].text.strip()
            return f"{columns[0].text}: ₹{price}" if lang == "en" else f"{columns[0].text} : ₹{price} (தமிழ்)"

    return "தேர்ந்தெடுத்த பயிருக்கு விலை தரவுகள் கிடைக்கவில்லை." if lang == "ta" else "No price data available for the selected crop."



@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip().lower()
    lang = data.get("language", "en")
    location = data.get("location")

    if not user_message:
        return jsonify({"response": translations[lang]["error"]})

    location_text = ""
    if location:
        lat, lon = location["latitude"], location["longitude"]
        location_text = f" (உங்கள் இருப்பிடம்: {get_location_info(lat, lon)})" if lang == "ta" else f" (User's location: {get_location_info(lat, lon)})"

        if "weather" in user_message.lower() or "வானிலை" in user_message:
            weather_info = get_weather(lat, lon, lang)
            return jsonify({"response": weather_info})

    if "market price" in user_message or "சந்தை விலை" in user_message:
        crop = user_message.split()[-1]  # Assume crop is the last word
        market_info = get_market_price(crop)
        return jsonify({"response": market_info})

    try:
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": user_message}])
        bot_reply = response["message"]["content"]
        return jsonify({"response": bot_reply + location_text})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)