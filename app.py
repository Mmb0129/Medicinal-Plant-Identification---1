# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template,redirect

# Configure GPU memory usage
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)



import openai
openai.api_key = '-7TG6s9ZMwQtKeIkZOeVkT3BlbkFJxyMqKSXwfHgzpIqMjDqw' #place your api key

def fetch_completion_from_conversations(conversations, model="gpt-3.5-turbo-0125", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=conversations,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def generate_summary(message):
    topic = message

    plants_list = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'ashoka', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'camphor', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'kamakasturi', 'Kambajala', 'Kasambruga', 'kepala', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomogranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric']

    conversations = [
        {'role': 'system',
        'content':"""Enter your query: """},
        {'role': 'user', 'content': f"cure for {message}"},
        {'role': 'system', 'content': f"""Please provide your query or concern so I can assist you with accurate information or remedies using medicinal plants from the list provided. {' '.join(plants_list)}"""}
    ]

    summary_content = fetch_completion_from_conversations(conversations, temperature=0.7, max_tokens=300)

    print("Response:")
    # print(summary_content)
    output_path = "summary_output.txt"
    write_file(output_path, summary_content)
    return summary_content
    



# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model_inception_final_1.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# Dictionary mapping class indices to plant names
"""
class_to_plant = {
    0: "Aloevera", 1: "Amla", 2: "Amruthaballi", 3: "Arali", 4: "Ashoka", 4: "Astma_weed",
    5: "Badipala", 6: "Balloon_Vine", 7: "Bamboo", 8: "Beans", 9: "Betel", 10: "Bhrami",
    11: "Bringaraja", 12: "Camphor", 13: "Caricature", 14: "Castor", 15: "Catharanthus",
    16: "Chakte", 17: "Chilly", 19: "Citron lime(herelikai)", 20: "Coffee", 21: "Common rue(naagdalli)",
    22: "Coriander", 23: "Curry", 24: "Doddpathre", 25: "Drumstick", 26: "Ekka", 27: "Eucalyptus",
    28: "Ganigale", 29: "Ganike", 30: "Gasagase", 31: "Ginger", 32: "Globe Amarnath", 33: "Guava",
    34: "Henna", 35: "Hibiscus", 36: "Honge", 37: "Insulin", 38: "Jackfruit", 39: "Jasmine",
    40: "Kamakasturi", 41: "Kambajala", 42: "Kasambruga", 43: "Kepala", 44: "Kohlrabi",
    45: "Lantana", 46: "Lemon", 47: "Lemongrass", 48: "Malabar Nut", 49: "Malabar Spinach",
    50: "Mango", 51: "Marigold", 52: "Mint", 53: "Neem", 54: "Nelavembu", 55: "Nerale",
    56: "Nooni", 57: "Onion", 58: "Padri", 59: "Palak(Spinach)", 60: "Papaya", 61: "Parijatha",
    62: "Pea", 63: "Pepper", 64: "Pomogranate", 65: "Pumpkin", 66: "Raddish", 67: "Rose",
    68: "Sampige", 69: "Sapota", 70: "Seethaashoka", 71: "Seethapala", 72: "Spinach1",
    73: "Tamarind", 74: "Taro", 75: "Tecoma", 76: "Thumbe", 77: "Tomato", 78: "Tulsi", 79: "Turmeric"
}"""


class_to_plant ={
    0: "Aloevera", 1: "Bamboo", 2: "Betel", 3: "Castor", 4: "Catharanthus", 5: "Coriender",
    6: "Curry", 7: "Doddpathre", 8: "Guava", 9: "Hibiscus", 10: "Honge",
    11: "Jackfruit", 12: "Lemon", 13: "Mint", 14: "Neem", 15: "Palak(Spinach)",
    16: "Papaya", 17: "Seethapala", 18: "Tamarind", 19 : "Tulsi"
    }


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255.0  # Scaling to the range [0, 1]
    x = np.expand_dims(x, axis=0)

    # Make predictions
    preds = model.predict(x)
    pred_index = np.argmax(preds, axis=1)

    # Use the predicted index to retrieve the corresponding plant name
    if pred_index[0] in class_to_plant:
        preds = class_to_plant[pred_index[0]]
        print(preds)
        # generate_summary("Fever cure")
    else:
        preds = "Unknown Plant"

    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to a temporary location
        temp_path = 'temp_image.jpg'
        f.save(temp_path)

        # Make prediction
        preds = model_predict(temp_path, model)
        os.remove(temp_path)  # Remove the temporary image file

        return preds

    return None

@app.route("/chat_bot",methods=["GET"])
def chat():
    if request.method=="GET":
        return render_template("chat_bot.html")
@app.route("/chat_result" ,methods=["POST"])
def chat_result():
    if request.method=="POST":
        query = request.form['userInput']
        r=generate_summary(query)
        return render_template("chat_bot.html",result=r)
  



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
