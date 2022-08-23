from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

dic = {0 : 'exterior', 1 : 'interior'}

model = load_model('imageclassifier_v5_3aug.h5')

model.make_predict_function()

def predict_label(img_path):
	img = cv2.imread(img_path)
	resize = tf.image.resize(img, (256,256))
	yhat = model.predict(np.expand_dims(resize/255, 0))
	if yhat < 0.5:
		return dic[0]
	else:
		return dic[1] 

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "just a simple app ..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)