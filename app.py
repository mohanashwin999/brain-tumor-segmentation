from flask import *
import SimpleITK as sitk 
import numpy as np

import os
import glob

from keras.models import load_model
from keras import backend as K

import base64, urllib
from io import BytesIO

import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from flask_ngrok import run_with_ngrok
matplotlib.use('Agg')

app = Flask(__name__)
run_with_ngrok(app)

    
def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection) / (K.sum(K.square(y_true),axis=-1) + K.sum(K.square(y_pred),axis=-1) + epsilon)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def sobel(img):
  img_sobelx = cv2.Sobel(img, cv2.CV_8U,1,0,ksize=3)
  img_sobely = cv2.Sobel(img, cv2.CV_8U,0,1,ksize=3)
  img_sobel = img_sobelx + img_sobely+img
  return img_sobel

def npy_to_img(numpy_img, name):
    path = os.path.dirname(os.path.abspath(__file__))+"\\static\\images\\"
    plt.figure(figsize=(192, 192))
    plt.imshow(numpy_img)
    plt.axis('off')
    fig = plt.gcf()
    fig.savefig(path+name+".png", format='png')


def predict_and_save_images(data):
    files = glob.glob(os.path.dirname(os.path.abspath(__file__))+"\\static\\images\\*")
    if files:
        for f in files:
            os.remove(f)

    npy_to_img(data[0,:,:,0], "flair")
    npy_to_img(data[0,:,:,1], "t1")
    npy_to_img(data[0,:,:,2], "t1ce")
    npy_to_img(data[0,:,:,3], "t2")
    
    path = os.path.dirname(os.path.abspath(__file__))+"\\models\\"
    sobel_unet_model = load_model(path+"sobel-with-unet-model.hdf5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef })
    vnet_model = load_model(path+"vnet-model.hdf5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef })
    wnet_model = load_model(path+"wnet-model.hdf5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef })
    unet_model = load_model(path+"unet-model.hdf5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef })
    
    sobel_data = sobel(data)
    data = (data-np.mean(data))/np.max(data)
    sobel_data= (sobel_data-np.mean(sobel_data))/np.max(sobel_data)

    sobel_unet_image = np.argmax(sobel_unet_model.predict(sobel_data),axis=-1)
    sobel_unet_image = sobel_unet_image.reshape(-1,192,192,1)

    unet_image = np.argmax(unet_model.predict(data),axis=-1)
    unet_image = unet_image.reshape(-1,192,192,1)

    vnet_image = np.argmax(vnet_model.predict(data),axis=-1)
    vnet_image = vnet_image.reshape(-1,192,192,1)

    wnet_image = np.argmax(wnet_model.predict(data),axis=-1)
    wnet_image = wnet_image.reshape(-1,192,192,1)

    npy_to_img(unet_image, "unet_image")
    npy_to_img(sobel_unet_image, "sobel_unet_image")
    npy_to_img(vnet_image, "vnet_image")
    npy_to_img(wnet_image, "wnet_image")

    


def load_data(flair, t1, t1ce, t2,sliceno):
  data = []
  img_itk = sitk.ReadImage(flair)
  flair = sitk.GetArrayFromImage(img_itk)
  img_itk = sitk.ReadImage(t1)    
  t1 =  sitk.GetArrayFromImage(img_itk)
  img_itk = sitk.ReadImage(t1ce)
  t1ce =  sitk.GetArrayFromImage(img_itk)
  img_itk = sitk.ReadImage(t2)
  t2 =  sitk.GetArrayFromImage(img_itk)
  data.append([flair,t1,t1ce,t2])
  data = np.asarray(data,dtype=np.float32)
  data = np.transpose(data,(0,2,3,4,1))
  data = data[:,sliceno,30:222,30:222,:].reshape([-1,192,192,4])
  return data

def return_path(str):
    return os.path.dirname(os.path.abspath(__file__))+"\\uploads\\{}.nii.gz".format(str)

@app.route("/", methods=['GET','POST'])
def index():
    if request.method == "POST":
        files = glob.glob(os.path.dirname(os.path.abspath(__file__))+"\\uploads\\*")
        if files:
            for f in files:
                os.remove(f)

        flair_path = return_path("flair")
        t1_path = return_path("t1")
        t1ce_path = return_path("t1ce")
        t2_path = return_path("t2")
        
        request.files['flair'].save(flair_path)
        request.files['t1'].save(t1_path)
        request.files['t1ce'].save(t1ce_path)
        request.files['t2'].save(t2_path)
        
        sliceno = int(request.form.get("sliceno"))

        data = load_data(flair_path, t1_path, t1ce_path, t2_path, sliceno)
        predict_and_save_images(data)

        files = glob.glob(os.path.dirname(os.path.abspath(__file__))+"\\uploads\\*")
        for f in files:
            os.remove(f)

        return render_template("output.html", sliceno = sliceno )

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)