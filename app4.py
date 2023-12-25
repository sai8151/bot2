from flask import Flask, request, jsonify
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load JSON data from train.json
with open('train.json', 'r') as json_file:
    investment_data = json.load(json_file)

# Extract plans and target audience, handling missing 'plan' key
plans = [plan.get('plan', '').lower() for plan in investment_data['schemes']]
target_audience = [plan.get('target_audience', '').lower()
                   for plan in investment_data['schemes']]

# Combine plan names and target audience for similarity analysis
combined_text = [f"{plan} {audience}" for plan,
                 audience in zip(plans, target_audience)]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined_text)


def get_most_similar_plan(user_input):
    # Process user input
    user_input = user_input.lower()

    # Transform user input using the same vectorizer
    user_tfidf = vectorizer.transform([user_input])

    # Calculate cosine similarity
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)[0]

    # Get the index of the most similar plan
    most_similar_index = similarities.argmax()

    # Return the most similar plan's message
    return f"Recommended Plan: {plans[most_similar_index]} - {target_audience[most_similar_index]}"


@app.route('/bot', methods=['POST'])
def bot():
    data = request.get_json()
    user_message = data.get('message', '')

    if user_message:
        response = get_most_similar_plan(user_message)
    else:
        response = "Error: No message provided."

    return jsonify({'message': response})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
