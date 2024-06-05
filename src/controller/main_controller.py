from flask import Blueprint, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from werkzeug.utils import secure_filename

# Importing necessary libraries
import nltk
import pandas as pd
import numpy as np
from textblob import Word
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

predict = Blueprint("predict", __name__, url_prefix="/ai")

model = load_model("src/model/model1.h5", compile=False)
stop_words = stopwords.words("english")


def cleaning(text):
    # Convert to lowercase
    text = " ".join(x.lower() for x in text.split())
    # Replacing the digits/numbers
    text = "".join([i for i in text if not i.isdigit()])
    # Removing stop words
    text = " ".join(x for x in text.split() if x not in stop_words)
    # Lemmatization
    text = " ".join([Word(x).lemmatize() for x in text.split()])
    return text


@predict.route("/sentiment", methods=["POST"])
def sentiment():
    try:
        data = request.get_json()
        text = data.get("text", "")
        text = cleaning(text)
        max_words = 1000
        max_len = 1008
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts([text])
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len)
        predictions = model.predict(padded_sequence)

        print(predictions)

        labels = ["POSITIVE", "NEURAL", "NEGATIVE"]
        predicted_labels = predictions.argmax(axis=-1)
        result = labels[predicted_labels[0]]
        return jsonify({"result": result})
    except Exception as e:
        print(e)
        return jsonify({"message": "An error occurred during prediction"}), 500
