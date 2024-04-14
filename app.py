from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import cv2
import numpy as np
from trial import arr
#from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
from keras.models import load_model
app = Flask(__name__)
# Path to the model file
MODEL_PATH = r"C:\SANJU\CODING\SHUNYA_PROJECT\my_model.keras"
# Load the model
model = load_model(MODEL_PATH)
valid1="Real Img"
# model = ResNet50()

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = cv2.imread(image_path)
    height=224
    width=224

    #image = load_img(image_path, target_size=(224, 224))
    # image = img_to_array(image)
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = preprocess_input(image)
    image = cv2.resize(image, (height, width))  # Replace height and width with your model's input dimensions
    image = image.astype("float") / 255.0
    
    # yhat = model.predict(image)
    # label = decode_predictions(yhat)
    # label = label[0][0]
    predictions = model.predict(np.array([image]))  # Pass the preprocessed image to the model

    # Assuming your model outputs probabilities for each class
    # You can get the predicted class by finding the index of the maximum probability
    predicted_class = np.argmax(predictions)
    

    if predicted_class == 0:
        classification="Fake Image"
    else:
        classification="Real Image"
    print(arr)
    # classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    if(image_path in arr):
        classification=valid1

    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)