from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import ImageFile,Image
import PIL.Image
from io import BytesIO
import fastai
import torch
from fastai import *
from fastai.vision import *
from typing import List
import sys
quality_model=load_model('local_rotten_lr2_final.h5')
clf_model = load_learner('./')


# reads from file object
# return array of original uploaded image and 1x100x100x3 processed image
def preprocess(inImg):
    ImageFile.LOAD_TRUNCATED_IMAGES =False
    org_img=PIL.Image.open(BytesIO(inImg.read()))
    org_img.load()
    img=org_img.resize((100,100), PIL.Image.ANTIALIAS)
    img=image.img_to_array(img)
    org_img=image.img_to_array(org_img)

    return org_img, np.expand_dims(img,axis=0)


# return [prob_for_fresh, prob_for_rotten]
def check_rotten(img):
    return [round(100*quality_model.predict(img)[0][0],3),round(100*(1-quality_model.predict(img)[0][0]),3)]

def get_topkaccuracy(pred_probs: List, n: int = 7) -> List[dict]:

    class_probs = []
    class_names = get_classnames()

    for i in range(len(pred_probs)):
        class_probs.append((class_names[i], pred_probs[i]))

    class_probs = sorted(class_probs, reverse=True, key=lambda x: x[1])

    top_k = {}
    for class_prob in class_probs[:n]:
      top_k[class_prob[0]]=round(class_prob[1] * 100, 2)

    return top_k

def get_classnames() -> List:

    class_names = clf_model.data.classes
    for i in range(len(class_names)):
        class_names[i] = class_names[i].replace("_", " ")
        class_names[i] = class_names[i].title()

    return class_names


def classify_fruit(img):
   pred_class,pred_idx,pred_probs = clf_model.predict(img)
   pred_probs = pred_probs / sum(pred_probs)
   pred_probs = pred_probs.tolist()

   top_k = get_topkaccuracy(pred_probs)

   return top_k
