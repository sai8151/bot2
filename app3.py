import re
import difflib
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Load your JSON data
with open('train.json', 'r') as json_file:
    investment_data = json.load(json_file)


@app.route('/bot', methods=['POST'])
def bot():
    data = request.get_json()
    message = data['message']

    # Check if the message is present in the investment data
    response = get_plan_info(message)

    if not response:
        response = generate_default_response()

    # Concatenate all values in the "plan" dictionary
    response_string = " ".join(str(value) for value in response.values())

    return jsonify({'message': response_string})


def get_plan_info(message):
    for plan in investment_data.get('schemes', []):
        if 'plan' in plan and message.lower() == plan['plan'].lower():
            return plan
    return None


def generate_default_response():
    return "Sorry, I couldn't find information for that query. Please try again."


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
