
########################### Code Description#############################
1. TRANSFER .ppm to .jpg:
PPMtoJPG_all.py #training dataset
dataset:
All training .ppm dataset

2. CREATE .lmdb
create_lmdb_all.py #training and testing datasets in training
dataset:
All training .jpg images

3. COMPUTE IMAGE MEAN
meanTrain.binaryproto #training
meanTest.bianaryproto #testing dataset in training
dataset:
test_lmdb folder
train_lmdb folder

4. Main script to train the CNN
CNN_Final.py
Net_Final43.prototxt
Solver_Final.prototxt
test_lmdb folder
train_lndb folder
