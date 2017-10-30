# load model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import caffe
import glob
import cv2
from caffe.proto import caffe_pb2


caffe.set_mode_gpu()

net = caffe.Net('Net_43Test.prototxt', # defines the structure of the model
                'Model_43class.caffemodel', # contains the trained weights
                caffe.TEST) # use test mode (e.g., don't perform dropout)

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization: is a technique for adjusting image intensities to enhance contrast.
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    # INTER_CUBIC: a bicubic interpolation over 4x4 pixel neighborhood
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    return img

# Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/sanwang/Desktop/FinalProject/GTSRB/meanFinalTest.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

# Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2, 0, 1))
'''
Making predicitions
'''
# Reading image paths
test_img_paths = [img_path for img_path in glob.glob("/home/sanwang/Desktop/FinalProject/data/Final_Test/JPG/*.jpg")]

# Making predictions
test_ids = ['Index']
preds = ['PredictClass']
LabelName = ['ClassName']
Prob1 = ['Prob']
preds2 = ['2ndClass']
LabelName2 = ['2ndClassName']
Prob2 = ['2ndProb']
preds3 = ['3rdClass']
LabelName3 = ['3rdClassName']
Prob3 = ['3rdProb']
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    transformed = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    plt.imshow(transformed)
    net.blobs['data'].data[...] = transformer.preprocess('data',transformed)
    output = net.forward()
    Prob = output['prob'][0]  # output prob for the first image in the batch
    LabelFile = 'Label.txt'
    label = np.loadtxt(LabelFile, str, delimiter='\t') # label file
    top_3 = Prob.argsort()[::-1][:3] # reverse sort and take five largest
    top1_class = Prob.argmax()
    top2_class = top_3[1]
    top3_class = top_3[2]
    top1_prob = Prob[top1_class]
    top2_prob = Prob[top2_class]
    top3_prob = Prob[top3_class]
    top1_label = label[top1_class]
    top2_label = label[top2_class]
    top3_label = label[top3_class]

    test_ids = test_ids + [img_path.split('/')[-1]] # image name
    preds = preds + [top1_class]
    LabelName = LabelName + [label[top1_class]]
    Prob1 = Prob1 + [top1_prob]
    preds2 = preds2 + [top2_class]
    LabelName2 = LabelName2 + [label[top2_class]]
    Prob2 = Prob2 + [top2_prob]
    preds3 = preds3 + [top3_class]
    LabelName3 = LabelName3 + [label[top3_class]]
    Prob3 = Prob3 + [top3_prob]

    print 'Prediction result:'
    print img_path
    print Prob.argmax()
    print label[Prob.argmax()]
    print '-------'


'''
Making file to save top3 prediction wiht corresponsding probablity
'''
os.system('rm -rf  ' + '/home/sanwang/Desktop/FinalProject/GTSRB/TestResult.csv')

with open("/home/sanwang/Desktop/FinalProject/GTSRB/TestResult.csv", "w") as f:
    #f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i]) + "," + str(preds[i]) + "," + str(LabelName[i])+ "," + str(Prob1[i]) +
                "," + str(preds2[i]) + "," + str(LabelName2[i])+ "," + str(Prob2[i]) +
                "," + str(preds3[i]) + "," + str(LabelName3[i])+ "," + str(Prob3[i]) +"\n")

f.close()