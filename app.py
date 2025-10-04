# --- Conceptual Backend Code (e.g., Flask, Python) ---
# You need to install the SDKs: pip install google-genai Pillow Flask-CORS
# You need to set your API key as an environment variable: GEMINI_API_KEY
import os
import json
import logging
from io import BytesIO

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from waitress import serve # Import the Waitress serve function

# Use the new, idiomatic Python SDK for Google GenAI
from google import genai
from google.genai.errors import APIError

# --- env & logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurosort-api")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # Log the failure before raising an error
    logger.error("GEMINI_API_KEY is not set. Please set the environment variable.")
    raise RuntimeError("GEMINI_API_KEY is not set")



app = Flask(__name__)
# ----------------------------------------------------------------------------------
# 🔑 IMPORTANT: Initialize CORS to allow requests from your frontend.
# Using origins='*' for development/conceptual testing. Replace with your actual frontend URL (e.g., 'https://your-domain.github.io')
# for production security.
CORS(app, 
     resources={r"/api/*": {"origins": "*"}},
     methods=["POST", "OPTIONS"],
     allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
)
# ----------------------------------------------------------------------------------
                                           
# Initialize the GenAI Client
client = genai.Client()

@app.route("/")
def home():
    """Simple status check endpoint."""
    return jsonify({"status": "ok", "service": "Gemini NeuroSort API"})


@app.route('/api/gemini-predict', methods=['POST','OPTIONS'])
def gemini_predict():

    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        return ('', 204)
        
    # 1. Input Validation
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    # Prompt is required for multimodal vision tasks
    prompt = request.form.get('prompt') # Get the prompt sent from the frontend
    
    if not prompt:
         return jsonify({"error": "No text prompt provided"}), 400
         
    try:    
        # 2. Convert the uploaded file data into a PIL Image object
        image_data = file.read()
        image = Image.open(BytesIO(image_data))
        
        logger.info(f"Received image and prompt: {prompt}. Sending request to Gemini...")
    
    
        # 3. Call the Gemini model
        # NOTE: For production, strongly consider using response_schema in generation_config
        # to force the model to return a structured JSON response instead of relying on
        # fragile string matching (like done below).
        model_response = client.models.generate_content(
            model='gemini-2.5-flash', # Or gemini-2.5-pro for more complex reasoning
            contents=[prompt, image]
        )
        
        # 4. Process the response (Crude String Parsing - See NOTE above)
        response_text = model_response.text        
        
        if "Recyclable" in response_text:
            category = "Recyclable"
        elif "Compost" in response_text:
            category = "Compost"
        else:
            category = "Non-Recyclable"
        
        # Log the result on the backend
        logger.info(f"Prediction complete. Category: {category}. Response snippet: {response_text[:50]}...")
        
        return jsonify({
            "success": True,
            "prediction": response_text,
            "category": category,
        })
        
        
    except APIError as e:
        logger.error(f"Gemini API Error: {e}")
        return jsonify({"error": f"Gemini API service failed: {e.message}"}), 500
    
    except Exception as e:
        logger.error(f"Internal Server Error: {e}")
        return jsonify({"error": "Internal AI prediction service failed."}), 500
        
if __name__ == '__main__':
    # Using Waitress for production-ready serving instead of Flask's built-in server.
    # The debug=True parameter is removed as Waitress handles serving, not debugging.
    # The server will still run on port 5000.
    logger.info("Starting Waitress server on http://0.0.0.0:5000...")
    serve(app, host='0.0.0.0', port=5000)
    
