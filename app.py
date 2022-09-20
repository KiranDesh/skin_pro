from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2

app = Flask(__name__)

dic = {0:'akiec', 1:'bcc', 2:'bkl', 3:'df', 4:'mel', 5:'nv', 6:'vasc'}

model = load_model('model_saved.h5')

model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path,target_size=(224,224))
	i = img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	pred = model.predict(i)
	classes = np.argmax(pred,axis=1)
	return dic[classes[0]]

#predict_x=model.predict(X_test) 
#classes_x=np.argmax(predict_x,axis=1)


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)
info_path = r'C:\Users\kdesh\Downloads\new_project_info.docx'

app.route("/submit",methods = ['GET','POST']
def fetch():
        for i in info_path:
                if i == 

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
