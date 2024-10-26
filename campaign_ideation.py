from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)
LLM_API_URL = "https://api.openai.com/v1/completions" 
API_KEY = os.getenv('API_KEY')

@app.route("/generate-campaign", methods=["POST"])
def generate_campaign():
    data = request.json

    # Extract input parameters
    campaign_type = data.get("campaignType")
    socio_economic_class = data.get("socioEconomicClass")
    product_type = data.get("productType")
    format_type = data.get("format")
    demographic = data.get("demographic")
    geographic = data.get("geographic")
    budget = data.get("budget")
    campaign_objective = data.get("campaignObjective")
    prompt = f"""
    Generate a series of creative marketing campaign ideas based on the following inputs:
    - Campaign Type: {campaign_type}
    - Socio-Economic Class: {socio_economic_class}
    - Product Type: {product_type}
    - Format: {format_type}
    - Demographic: {demographic}
    - Geographic: {geographic}
    - Budget: {budget}
    - Campaign Objective: {campaign_objective}

    For each campaign, provide the following in html string wihout head and body tag:
    1. Campaign Theme & Slogan: A central concept that resonates with the demographic, along with a catchy slogan.
    2. Messaging Variations: Suggestions for tailored messages to appeal to different segments within the demographic.
    3. Strategies & Channels: Recommendations for channels (e.g., social media, email marketing, influencers) and creative strategies suited to the objectives for the {product_type}.
    4. Campaign Calendar: Suggested timing for key actions and events.
    5. Sentiment Analysis Predictions: Expected audience reactions and suggestions to optimize positive engagement.

    Output this in a format tailored to marketing teams, providing clear and actionable insights for implementing the campaign ideas.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": prompt,
        "max_tokens": 1500,
        "temperature":0.7,
        "presence_penalty":0.6
    }
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        print(result)
        campaign_ideas = result["choices"][0]["text"]


        return ({"campaignIdeas": campaign_ideas})

    except requests.exceptions.RequestException as e:
        print(e)
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
