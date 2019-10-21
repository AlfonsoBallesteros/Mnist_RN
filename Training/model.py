from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt

def clasificar(filename):
    # load model
    model = load_model('../Models/model.h5')

    imagen = plt.imread('../Uploads/' + filename)
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

clasificar('../Image/tres.png')
