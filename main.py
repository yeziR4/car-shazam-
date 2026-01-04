import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# --- KEYS FROM RENDER ENV VARS ---
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

def get_car_identification(image_url):
    """Google Lens for visual identification."""
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

def generate_report_with_groq(lens_data, country):
    """Llama 3.3 70B research and formatting."""
    # Extract titles from Lens
    matches = [m.get("title") for m in lens_data.get("visual_matches", [])[:3]]
    
    # SYSTEM PROMPT: Set the persona and output format
    system_message = "You are an expert automotive researcher. You must respond ONLY with a valid JSON object."
    
    # USER PROMPT: The specific task
    user_prompt = f"""
    The car visual matches are: {matches}.
    
    TASK:
    1. Identify the Brand, Model, and Year range.
    2. Provide 2026 technical specs (Engine, HP, Transmission, Drivetrain) using your internal knowledge.
    3. Suggest 3 real URLs for car inventory in {country}.
    
    REQUIRED JSON STRUCTURE:
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
      "confidence_score": 95
    }}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # FORCES JSON OUTPUT
            temperature=0.2, # Lower temperature for more factual specs
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"error": f"Groq research failed: {str(e)}"}

@app.route('/identify', methods=['POST'])
def identify():
    data = request.json
    image_url = data.get("image_url")
    country = data.get("country", "United States")
    
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400
    
    # 1. Identify visually
    lens_results = get_car_identification(image_url)
    if not lens_results:
        return jsonify({"error": "Failed to identify image"}), 500
        
    # 2. Generate report using Llama 3.3
    report = generate_report_with_groq(lens_results, country)
    
    # 3. Final error handling for frontend safety
    if "error" in report:
        return jsonify(report), 500
        
    return jsonify(report)

@app.route('/')
def health_check():
    return "Car Shazam API (Groq Version) is Online!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
