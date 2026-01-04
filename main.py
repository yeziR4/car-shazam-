import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
# Enable CORS so your frontend can access this API
CORS(app)

# Pulling keys from Render Environment Variables
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

def get_car_identification(image_url):
    """Identifies the car via Google Lens."""
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_KEY,
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        return response.json()
    except Exception as e:
        print(f"Lens Error: {e}")
        return None

def generate_report(lens_data, country):
    """Generates the final JSON report using Gemini with search grounding."""
    # Grab the top visual matches from Lens
    matches = [m.get("title") for m in lens_data.get("visual_matches", [])[:3]]
    
    prompt = f"""
    The car in the image is one of these: {matches}.
    
    TASK:
    1. Identify the specific Brand and Model.
    2. Provide the key technical specs (Engine, HP, Transmission, Drivetrain) for the 2026 model using your INTERNAL KNOWLEDGE.
    3. SEARCH the web specifically for 3 CURRENT 2026 model listings or inventory pages in {country}.
    
    Return ONLY a JSON object:
    {{
      "identification": {{
        "brand": "string",
        "model": "string",
        "year_range": "2024-2026"
      }},
      "key_specifications": {{
        "engine": "string",
        "horsepower": "string",
        "transmission": "string",
        "drivetrain": "string"
      }},
      "local_market": {{
        "country": "{country}",
        "live_listings": ["url1", "url2", "url3"],
        "summary": "Brief summary of 2026 availability"
      }},
      "confidence_score": 0-100
    }}
    """
    try:
        # Enable search tool only for listings
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{'google_search': {}}], 
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"Gemini Error: {str(e)}"}

@app.route('/identify', methods=['POST'])
def identify_car():
    data = request.json
    image_url = data.get("image_url")
    country = data.get("country", "United States")
    
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400
    
    # Run the Shazam process
    lens_results = get_car_identification(image_url)
    if not lens_results:
        return jsonify({"error": "Could not identify image"}), 500
        
    report = generate_report(lens_results, country)
    return jsonify(report)

if __name__ == "__main__":
    # Render provides the PORT env var automatically
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
