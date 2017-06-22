############################################CREATE TRANING###############################################################import os
import os
import caffe
import matplotlib.pyplot as plt
import numpy as np
import numpy
############################################Pre Processing Data##############################################################
my_root = 'path'
os.chdir(my_root)


#######################################Train the Network with the Solver######################################################

caffe.set_device(0)
caffe.set_mode_gpu()

# Use SGDSolver, namely stochastic gradient descent algorithm
solver = caffe.SGDSolver('Solver_Final.prototxt')
#----------need to run the following command to gpustat works-------------------------------
os.system("gpustat")
#----------------------------------------------------------------------------------------------
#---------------------------------------Training Caffe-----------------------------------------

niter =1000
test_interval = 50
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')

    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc


################## save trained net #############
solver.net.save('Model_43class.caffemodel')

#----------------------------------------------------------------------------------------------
###########################Plotting Intermediate Layers, Weight################################
#---------------------------------------Define Functions---------------------------------------
# vis_square_f is for plotting feature maps
def vis_square_f(data):
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data);
    plt.axis('off')
#----------------------------------------------------------------------------------------------
#------------------------------Test Accuracy & Loss--------------------------------------------
plt.figure(1)
plt.semilogy(np.arange(niter), train_loss)
plt.xlabel('Number of Iteration')
plt.ylabel('Training Loss Values')
plt.title('Training Loss')

plt.figure(2)
plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy Values')
plt.title('Test Accuracy')

net = solver.net

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#------------------------------Plot All Kernels for Conv1(github feature extract)---------------------------------------
nrows = 4                                   # Number of Rows
ncols = 5                                   # Number of Columbs
ker_size = 3                                # Kernel Size
Zero_c= np.zeros((ker_size,1))              # Create np.array of zeros
Zero_r = np.zeros((1,ker_size+1))
M= np.array([]).reshape(0,ncols*(ker_size+1))

for i in range(nrows):
    N = np.array([]).reshape((ker_size+1),0)

    for j in range(ncols):
        All_kernel = net.params['conv1'][0].data[j + i * ncols][0]

        All_kernel = numpy.matrix(All_kernel)
        All_kernel = np.concatenate((All_kernel,Zero_c),axis=1)
        All_kernel = np.concatenate((All_kernel, Zero_r), axis=0)
        N = np.concatenate((N,All_kernel),axis=1)
    M = np.concatenate((M,N),axis=0)

plt.figure(3)
plt.imshow(M, cmap='Greys',  interpolation='nearest')
plt.title('All Kernels for Conv1')

#----------------------------------------------------------------------------------------------
#------------------------------FM for conv1(github feature extract)---------------------------------------
FM1 = net.blobs['conv1'].data[0, :20]
plt.figure(4)
vis_square_f(FM1)
plt.title('Feature maps for conv1')
#----------------------------------------------------------------------------------------------
#------------------------------pooling1 (github feature extract)---------------------------------------
pool1 = net.blobs['pool1'].data[0]
plt.figure(5)
vis_square_f(pool1)
plt.title('output of pooling1')
#----------------------------------------------------------------------------------------------
#------------------------------Plot All Kernels for Conv2---------------------------------------
nrows = 4                                   # Number of Rows
ncols = 8                                   # Number of Columbs
ker_size = 3                                # Kernel Size
Zero_c= np.zeros((ker_size,1))              # Create np.array of zeros
Zero_r = np.zeros((1,ker_size+1))
M= np.array([]).reshape(0,ncols*(ker_size+1))

for i in range(nrows):
    N = np.array([]).reshape((ker_size+1),0)

    for j in range(ncols):
        All_kernel = net.params['conv2'][0].data[j + i * ncols][0]

        All_kernel = numpy.matrix(All_kernel)
        All_kernel = np.concatenate((All_kernel,Zero_c),axis=1)
        All_kernel = np.concatenate((All_kernel, Zero_r), axis=0)
        N = np.concatenate((N,All_kernel),axis=1)
    M = np.concatenate((M,N),axis=0)

plt.figure(4)
plt.imshow(M, cmap='Greys',  interpolation='nearest')
plt.title('All Kernels for Conv2')

#----------------------------------------------------------------------------------------------
#------------------------------FM for conv2 (github feature extract)---------------------------------------
FM2 = net.blobs['conv2'].data[0, :32]
plt.figure(7)
vis_square_f(FM2)
plt.title('Feature maps for conv2')
#----------------------------------------------------------------------------------------------
#------------------------------Pooling2 (github feature extract)---------------------------------------
pool2 = net.blobs['pool2'].data[0]
plt.figure(8)
vis_square_f(pool2)
plt.title('output of pooling2')
#----------------------------------------------------------------------------------------------
#------------------------------Plot All Kernels for Conv3 ---------------------------------------
nrows = 8                                   # Number of Rows
ncols = 8                                   # Number of Columbs
ker_size = 3                                # Kernel Size
Zero_c= np.zeros((ker_size,1))              # Create np.array of zeros
Zero_r = np.zeros((1,ker_size+1))
M= np.array([]).reshape(0,ncols*(ker_size+1))

for i in range(nrows):
    N = np.array([]).reshape((ker_size+1),0)

    for j in range(ncols):
        All_kernel = net.params['conv3'][0].data[j + i * ncols][0]

        All_kernel = numpy.matrix(All_kernel)
        All_kernel = np.concatenate((All_kernel,Zero_c),axis=1)
        All_kernel = np.concatenate((All_kernel, Zero_r), axis=0)
        N = np.concatenate((N,All_kernel),axis=1)
    M = np.concatenate((M,N),axis=0)

plt.figure(9)
plt.imshow(M, cmap='Greys',  interpolation='nearest')
plt.title('All Kernels for Conv3')

#----------------------------------------------------------------------------------------------
#------------------------------FM for conv3 (github feature extract)---------------------------------------
FM3 = net.blobs['conv3'].data[0, :64]
plt.figure(10)
vis_square_f(FM3)
plt.title('Feature maps for conv3')
#----------------------------------------------------------------------------------------------
#------------------------------pooling3 (github feature extract)---------------------------------------
pool3 = net.blobs['pool3'].data[0]
plt.figure(11)
vis_square_f(pool3)
plt.title('output of pooling3')
#----------------------------------------------------------------------------------------------
#------------------------------final probablity output(github feature extract)---------------------------------------
prob = net.blobs['prob'].data[0]
print "prob:", prob
plt.figure(12)
plt.plot(prob.flat)
plt.title('probablity output')
plt.show()
#----------------------------------------------------------------------------------------------
#---------------------------Print Shape ans Sizes for all Layers--------------------------------

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)






