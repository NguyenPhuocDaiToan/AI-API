from flask import Blueprint, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from werkzeug.utils import secure_filename

predict = Blueprint('predict', __name__, url_prefix='/predict')

model = load_model("src/model/model15.h5", compile=False)

def prepare_image(img_path, target_size=(299, 299)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@predict.route('/', methods=['POST'])
def predict_image():
    try:
        if 'img' not in request.files:
            return jsonify({"message": "No file part"}), 400
        
        file = request.files['img']
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('src/static/uploads', filename)
            file.save(file_path)

            img_array = prepare_image(file_path)
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            name_class = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            result = name_class[predicted_class_index]

            # Xóa tệp sau khi dự đoán
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return jsonify({"result": result})
    except Exception as e:
        print(e)
        return jsonify({"message": "An error occurred during prediction"}), 500
