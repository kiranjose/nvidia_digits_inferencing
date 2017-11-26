import numpy as np
import matplotlib.pyplot as plt
import caffe
import time
import cv2
cap = cv2.VideoCapture(0)
from skimage import io

MODEL_FILE = './deploy.prototxt'
PRETRAINED = './snapshot_iter_4864.caffemodel'
MEAN_IMAGE = './mean.jpg'
#Caffe
mean_image = caffe.io.load_image(MEAN_IMAGE)
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
channel_swap=(2,1,0),
raw_scale=255,
image_dims=(256, 256))
#OpenCv loop
while(True):
    start = time.time()
    ret, frame = cap.read()
    resized_image = cv2.resize(frame, (256, 256)) 
    cv2.imwrite("frame.jpg", resized_image)
    IMAGE_FILE = './frame.jpg'
    im2 = caffe.io.load_image(IMAGE_FILE)
    inferImg = im2 - mean_image
    #print ("Shape------->",inferImg.shape)
    #Inferencing
    prediction = net.predict([inferImg])
    end = time.time()
    pred=prediction[0].argmax()
    #print ("prediction -> ",prediction[0]) 
    if pred == 0:
       print("cat")
    else:
       print("dog")
    #Opencv display
    cv2.imshow('frame',inferImg)
    cv2.imshow('frame2',im2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
