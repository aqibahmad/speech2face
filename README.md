# Speech2Face 

### *Aqib Ahmad, Taimoor Aftab, Syed Haider Bokhari, Dr. Omer Ishaq*
#### FYP Project for Spring 2020 at FAST-NUCES, Islamabad

---



An implemention of the following papers:
> 1) Speech2Face: Learning the Face Behind a Voice 
   (Tae-Hyun Oh, Tali Dekel, Changil Kim, Inbar Mosseri, William T. Freeman, Michael Rubinstein, Wojciech Matusik)
   CVPR 2019

> 2) Synthesizing Normalized Faces from Facial Identity Features
   (Forrester Cole, David Belanger, Dilip Krishnan, Aaron Sarna, Inbar Mosseri, William T. Freeman)
   CVPR 2017



The repository includes the following code:
1) Scripts for data preprocessing for the facial decoder and the voice encoder models
2) PyTorch models for Facial Encoder (VGG-face recognition), Facial Decoder and Voice Encoder
3) Flask Server to deploy all these models 
4) Links to datasets for Facial Decoder and Voice Encoder
5) Python Notebooks for training the Facial Deocoder and Voice Encoder models


References: 
> Face Morphing Library: https://github.com/alyssaq/face_morpher
> Data pre-processing for Voice Encoder: https://github.com/saiteja-talluri/Speech2Face


speech recognition based on facial images

The project consists of 2 major models:
1) Sound to FaceVector: converts soundwave into a facial recognition vector
2) FaceVector to Image: converts the above mentioned vector to an image

Current implementation consists of FaceVector to Image model

INSTRUCTIONS:

1) Upload notebook onto Google Drive
2) For VGG-16 backend, make sure you get at least 10GB of CUDA memory
3) For Facenet backend, any graphics card on Colab will suffice
4) Connect to Google Drive

TEST INSTRUCTIONS:

1) Run the cells containing imports, model classes and model loading
2) Upload test images
3) Run the cell for testing

TRAIN INSTRUCTIONS:

1) Download the required batches from Google Drive
2) Specify required learning rate and interations
3) Load pre-saved model
4) Select Run All from google colab


