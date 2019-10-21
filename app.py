import os
#import magic
import urllib.request
#from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import keras
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt

UPLOAD_FOLDER = './Uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
			x = clasificar(filename)
			return "{'name': "+ str(x)+"}"
		else:
			flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
			return redirect(request.url)

#@app.route('/clasificar', methods=['GET'])   
def clasificar(filename):
    # load mod
    model = load_model('./Models/model.h5')
    print('modelo ')
    imagen = plt.imread('./Uploads/' + filename)
    #imagen = plt.imread('uploads/tres.png')
    #plt.imshow(imagen)
    #plt.show()
    print(imagen.shape)
    escala_grises = imagen[:,:,0]
    #plt.imshow(escala_grises)
    #plt.show()
    rezise = np.reshape(escala_grises, (1,28,28,1))
    predic = model.predict(rezise)
    print(predic)
    resultado = np.argmax(predic)
    print(resultado)

    return(resultado)

if __name__ == "__main__":
    app.run()
