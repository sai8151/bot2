import re
import difflib
import requests
import gc
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup, Comment
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Use a smaller model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Explicitly release resources after each request

# Load your JSON data
with open('train.json', 'r') as json_file:
    investment_data = json.load(json_file)


@app.before_request
def before_request():
    pass


@app.after_request
def after_request(response):
    # Explicitly release resources
    gc.collect()
    return response


def fetch_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def scrape_website(url):
    try:
        scraped_text = fetch_website(url)
        soup = BeautifulSoup(scraped_text, 'html.parser')

        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)

        # Extract the visible text from the webpage
        text = u" ".join(t.strip() for t in visible_texts)
        return text

    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""


def get_plan_info(plan_name):
    # Check if the plan is present in the investment data
    for plan in investment_data['schemes']:
        if 'plan' in plan and plan_name.lower() == plan['plan'].lower():
            return plan
    return None


def generate_response(input_text, max_length=500):
    # check if the input text is a URL
    if 'http' in input_text or 'www' in input_text:
        scraped_text = scrape_website(input_text)
        # Limit the scraped text to a manageable length
        input_text += " " + scraped_text[:500]

    # encode the input text
    input_text = tokenizer.encode(input_text, return_tensors='pt')

    # generate a response with variable length
    response = model.generate(
        input_text,
        max_length=max_length,
        temperature=0.7,  # Adjust the temperature
        num_return_sequences=3,
        top_k=60,
        top_p=0.6,
        no_repeat_ngram_size=2,
        num_beams=5,
    )

    # decode the response
    response_text = tokenizer.decode(response[0])

    return response_text


def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def post_process_response(response_text, max_sentences=3, similarity_threshold=0.7):
    """Removes duplicate sentences from the response text, considering similarity."""

    # Normalize punctuation for consistency
    response_text = re.sub(r"[.,!?]+", ".", response_text)

    # Split text into sentences using a more comprehensive pattern
    sentences = re.split(
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", response_text)

    # Initialize a set to efficiently store unique sentences
    unique_sentences = set()

    # Prioritize longer sentences and maintain original order
    filtered_response = ""
    for sentence in sorted(sentences, key=len, reverse=True):
        # Normalize for case-insensitive comparison
        normalized_sentence = sentence.strip().lower()

        # Check similarity with existing unique sentences
        is_unique = all(similar(sentence, existing) <
                        similarity_threshold for existing in unique_sentences)

        if is_unique:
            unique_sentences.add(normalized_sentence)
            filtered_response += sentence + " "  # Concatenate the filtered sentence

            # Check if the maximum number of sentences is reached
            if len(unique_sentences) >= max_sentences:
                break

    return filtered_response.strip()


@app.route('/bot', methods=['POST'])
def bot():
    data = request.get_json()
    message = data['message']

    # Check if the message is present in the investment data
    plan_info = get_plan_info(message)
    if plan_info:
        response = plan_info
    else:
        # If not found in JSON data, proceed with web scraping
        if 'http' in message or 'www' in message:
            scraped_text = scrape_website(message)
            # Limit the scraped text to a manageable length
            message += " " + scraped_text[:500]

        response = generate_response(message, max_length=800)

    print("Response:", response)  # Add this line for debugging
    return jsonify({'message': response})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
