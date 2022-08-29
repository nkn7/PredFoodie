import os
import sys
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image, ImageFile
import my_tf_mod
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import sys
import os
import fastai
from fastai import *
from fastai.vision import *
import fastai
os.environ['KMP_DUPLICATE_LIB_OK']='True'
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['GET','POST'])
def pred():
    if request.method=='POST':
         file = request.files['file']
         org_img, imgR= my_tf_mod.preprocess(file)
    
         classify_dict=my_tf_mod.classify_fruit(fastai.vision.open_image(file))
         rotten=my_tf_mod.check_rotten(imgR)
         print(classify_dict,flush=True)
         img_x=BytesIO()
         plt.imshow(org_img/255.0)
         plt.savefig(img_x,format='png')
         plt.close()
         img_x.seek(0)
         plot_url=base64.b64encode(img_x.getvalue()).decode('utf8')



    return render_template('Pred3.html', classify_dict=classify_dict,rotten=rotten, plot_url=plot_url)





if __name__=='__main__':
    app.run(debug=True)
