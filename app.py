from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

dic = {0: "exterior", 1: "interior"}

model = load_model('imageclassifier_v5.h5')

#why this bullshit?
model.make_predict_function()

def make_prediction(img_path):
    img = cv2.imread(img_path)
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    if yhat < 0.5 :
        return yhat, dic[0]
    else:
        return yhat, dic[1] 

#routes:
@app.route('/', methods = ["GET", "POST"])
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return "made in morocco baby!"

@app.route('/submit', methods = ["GET", "POST"])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)

        y, p  = make_prediction(img_path)
    return render_template("index.html", prediction = p, propa = y, img_path = img_path)

if __name__ == '__main__':
    app.run(debug=True)