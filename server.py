"""

Python Imports

Flask, PyTorch and repository classes

"""


import time
import os
import pickle


from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug import secure_filename

import numpy as np
import cv2
import torch


import torch.nn as nn
import torch.nn.functional as F
import torchfile

from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.vgg_face import VGG_16
from models.decoder import Decoder
from models.voice_encoder import VoiceEncoder
from preprocess.speaker import Speaker





"""

Load VGG Face Model

"""


encoder = VGG_16()
encoder.load_weights()
encoder.eval()

for p in encoder.parameters():
   p.requires_grad = False


"""

Load Facial Decoder 

"""


batchSize=110 
net = Decoder(batchSize)
checkpoint = torch.load("./weights/decoder-iter-4449.pt", map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net_state_dict'])
net.eval()


"""

Load Voice Encoder 

"""

x = Speaker()



net2 = VoiceEncoder(1)
checkpoint = torch.load("./weights/voice-encoder-epoch-16.pt", map_location=torch.device('cpu'))
net2.load_state_dict(checkpoint['net_state_dict'])
net2.eval()


"""

Load FaceMorpher repository

"""

os.environ["DLIB_DATA_DIR"]="./weights"
from face_morpher import facemorpher



"""

Functions for image input and output from Decoder

"""

def crop_and_align_img(input_img):
  src_points = facemorpher.locator.face_points(input_img)
  output_img = facemorpher.averager(input_img,src_points,src_points)
  return output_img


def get_output(picture_name, face_detect=False):

  net.eval()
  img_input_path = os.path.join(app.config['INPUT_FOLDER'], picture_name) 
  input = cv2.imread(img_input_path)


  if face_detect:
      input = crop_and_align_img(input)

  cv2.imwrite(img_input_path,input)
  output = net.test(input,encoder)
  img_output_path = os.path.join(app.config['OUTPUT_FOLDER'], picture_name) 
  cv2.imwrite(img_output_path,output)



def getSpectrogram(spect_path):
    spectrogram1 = np.zeros((1, 598, 257, 2))

    with open(spect_path, 'rb') as f:
      spectrogram1[0] = pickle.load(f)

    spectrogram1 = torch.Tensor(spectrogram1).permute(0,3,1,2)
  
    return spectrogram1


def get_output2(file_name):
  


  x.extract_wav(file_name)

  net.eval()
  img_input_path = os.path.join(app.config['SPECT_INPUT_FOLDER'], file_name+".pkl") 

  input = getSpectrogram(img_input_path)
  pred = net2.forward(input)


  output = net.forward_test(pred)

  img_output_path = os.path.join(app.config['OUTPUT_FOLDER'], file_name+".jpg") 
  cv2.imwrite(img_output_path,output)



"""

Flask Server

"""


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


app.config['INPUT_FOLDER'] = "input_imgs/"
app.config['OUTPUT_FOLDER'] = "output_imgs/"

app.config['SOUND_INPUT_FOLDER'] = "preprocess/data/audios/"
app.config['SPECT_INPUT_FOLDER'] = "preprocess/data/audio_spectrograms/"




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':

      if 'file' not in request.files:
          flash('No image selected')
          return redirect(request.url)

      f = request.files['file']

      if f.filename == '':
          flash('No image selected')
          return redirect(request.url)

      fn = secure_filename(f.filename)
      img_input_path = os.path.join(app.config['INPUT_FOLDER'], fn) 
      f.save(img_input_path)
      
      get_output(fn, face_detect=True)
      img_output_path = os.path.join(app.config['OUTPUT_FOLDER'], fn) 

      return render_template('predict.html', input_img=img_input_path,output_img=img_output_path)


@app.route('/predict2', methods = ['GET', 'POST'])
def predict2():
   if request.method == 'POST':

      if 'file' not in request.files:
          flash('No image selected')
          return redirect(request.url)

      f = request.files['file']

      if f.filename == '':
          flash('No image selected')
          return redirect(request.url)

      fn = secure_filename(f.filename)
      img_input_path = os.path.join(app.config['SOUND_INPUT_FOLDER'], fn) 
      f.save(img_input_path)

      fn = os.path.splitext(fn)[0]

      
      get_output2(fn)

      img_output_path = os.path.join(app.config['OUTPUT_FOLDER'], fn+".jpg") 

      return render_template('predict2.html', output_img=img_output_path)


@app.route('/input_imgs/<filename>')
def input_path(filename):
    return send_from_directory(app.config['INPUT_FOLDER'],
                               filename)

@app.route('/output_imgs/<filename>')
def output_path(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'],
                               filename)

@app.route('/imgs/<filename>')
def img(filename):
    return send_from_directory("imgs/",
                               filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)





#get_output("input_imgs/m3.png")