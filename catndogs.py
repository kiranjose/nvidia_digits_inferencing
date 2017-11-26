import numpy as np
import matplotlib.pyplot as plt
import caffe
import time
from PIL import Image

MODEL_FILE = '/home/catsndogs/deploy.prototxt'
PRETRAINED = '/home/catsndogs/snapshot_iter_480.caffemodel'
MEAN_IMAGE = '/home/catsndogs/mean.jpg'
# load the mean image
mean_image = caffe.io.load_image(MEAN_IMAGE)
#input the image file need to be tested
IMAGE_FILE = '/home/catsndogs/image_to_test.jpg'
im1 = Image.open(IMAGE_FILE)
# Tell Caffe to use the GPU
caffe.set_mode_gpu()
# Initialize the Caffe model using the model trained in DIGITS
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
channel_swap=(2,1,0),
raw_scale=255,
image_dims=(256, 256))
# Load the input image into a numpy array and display it
plt.imshow(im1)
# Iterate over each grid square using the model to make a class prediction
start = time.time()
inferImg = im1.resize((256, 256), Image.NEAREST)
inferImg -= mean_image
prediction = net.predict([inferImg])
end = time.time()
print(prediction[0].argmax())
pred=prediction[0].argmax()
if pred == 0: 
  print("cat")
else: 
  print("dog")
# Display total time to perform inference
print 'Total inference time: ' + str(end-start) + ' seconds'

