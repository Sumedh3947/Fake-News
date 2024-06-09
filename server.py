from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the saved model and vectorizer
with open('fake_news_detector.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Remove all extra whitespaces
    text = text.lower()  # Convert text to lowercase
    text = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    return ' '.join(text)


# Function to predict if the news is real or fake
def predict_news(news_text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(news_text)

    # Transform the preprocessed text using the TF-IDF vectorizer
    transformed_text = vectorizer.transform([preprocessed_text])

    # Make a prediction
    prediction = model.predict(transformed_text)

    # Return the prediction
    return 'Real' if prediction[0] == 1 else 'Fake'


@app.route("/")
def my_home():
    return render_template('index.html')


@app.route("/<string:page_name>")
def html_page(page_name):
    return render_template(page_name)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        news_text = data['news_text']
        prediction = predict_news(news_text)
        return jsonify(prediction)
    else:
        return 'Something went wrong. Please try again.'


if __name__ == '__main__':
    app.run(debug=True)
