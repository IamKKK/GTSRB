# GTSRB

### Code description
##### 1. TRANSFER .ppm to .jpg: 
PPMtoJPG_all.py #training dataset 
dataset: 
All training .ppm dataset 

##### 2. CREATE .lmdb 
create_lmdb_all.py #training and testing datasets in training 
dataset: 
All training .jpg images 

##### 3. COMPUTE IMAGE MEAN 
meanTrain.binaryproto 
meanTest.bianaryproto 
dataset: 
test_lmdb folder 
train_lmdb folder 

##### 4. Main script to train the CNN 
CNN_Final.py 
Net_Final43.prototxt 
Solver_Final.prototxt 
dataset: 
test_lmdb folder 
train_lndb folder 

### More description can be found in my blog:
[German Traffic Sign Recognition Benchmark blog](https://san-wang.github.io/blog/GTSRB/)

