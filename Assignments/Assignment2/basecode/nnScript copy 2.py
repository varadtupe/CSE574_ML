import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pandas as pd
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    res = 1.0 / (1.0 + np.exp(-1.0 * z))

    return res # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            print("lablelShape",label)
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    # Check variance of each col
    varCol = np.var(train_data,axis=0)
    indxDel = list(*np.where(varCol == 0))
    
    train_data = np.delete(train_data, indxDel, axis=1)
    validation_data = np.delete(validation_data, indxDel, axis=1)
    test_data = np.delete(test_data, indxDel, axis=1)
    

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    #print(w1.shape,w2.shape)

    # Your code here
    #
    #
    #
    #
    #
    dataLen = training_data.shape[0] # 50000
    
    # Forward propogation
    
    # Introducing bias
    inpDataB = np.c_[np.ones(len(training_data)),training_data]  #50000 x col+1
    
    # Equation 1: forward to hidden layer 1
    ly1Op=np.dot(inpDataB,np.transpose(w1)) #50000 x col+1 * col+1 x 50   = 50000 x 50
    
    # Equation 2: Activation function for hidden layer 1
    l1Act = sigmoid(ly1Op) # 50000 x 50
    
    # Introducing bias for hidden layer
    l1Bias = np.c_[np.ones(len(l1Act)),l1Act] # 50000 x 51
    
    # Equation 3: Propogate to output layer
    opLy = np.dot(l1Bias,np.transpose(w2)) # 50000 x 51 * 51 x 10 = 50000 x 10

    # Equation 4: Activation function for output layer
    op = sigmoid(opLy) # 50000 x 10
    
    # Label transpose
    #lbl = training_label.transpose().reshape(1,dataLen)
    lbl = training_label.astype(int)

    valLbl = lbl.transpose().reshape(dataLen,1)
    #print("valLbl",valLbl.shape) # 50000 x 1

    lbl = np.zeros((dataLen,n_class))

    for i in range(dataLen):
        index = int(training_label[i])
        lbl[i][index] = 1


    lblT = lbl.transpose() # 50000 x 10

    #print("lbl",lblT.shape)
    #print("op",op.shape) # 50000 x 10
    # print("mul", np.dot(lbl,np.log(op)))
    
    # Equation 5: Negative log likelyhood error ## Eq 6 & 7 combine

    #nllPre = np.dot(lblT , np.log(op)) + np.dot((1.0-lblT) , np.log(1.0-op))
    #nll = (-1/dataLen)*(np.sum(nllPre))

    nll = -np.sum(np.sum(np.matmul(lblT,np.log(op))+np.matmul((1.0-lblT),np.log(1.0-op)),1))/dataLen

    #print("nnl ",nll)
    #print("nnl ",nll.shape)
    #nll = nll.reshape(1,n_class)
    #print("nnl reshape",nll.shape)
    # Equation 8 & 9: Error delta & w2 Error
    delta = op - lbl
    
    w2Err = np.dot(np.transpose(delta),l1Bias)
    
    # Equation 10, 11, 12: w1 Error
    #print("w2Err",w2Err.shape)

    #print("l1Act",l1Act.shape)
    #print("np.dot(delta,w2)", np.dot(delta,w2).shape)
    #print("trin", training_data.shape)
    # print("temp",(1- l1Act.transpose())* l1Act.transpose())
    #print("temp1",(l1Act*(1.0- l1Act)).shape)
    #print("w2.shape",w2.shape)
    #print("delta",delta.shape)
    
    #w1Err = np.dot(np.dot(((1- l1Act.transpose())* l1Act.transpose()), np.dot(delta,w2)),training_data.transpose())
    #w1Err = np.dot(np.transpose(np.dot(np.dot(delta,w2).transpose(),(l1Act*(1.0- l1Act)))),training_data)

    s1 = np.multiply(l1Bias,(1.0 - l1Bias))
    #print("s1",s1.shape)
    s2 = np.dot(delta,w2)
    #print("s2",s2.shape)
    s3 = np.multiply(s2,s1)
    #print("s3",s3.shape)
    w1Err = np.dot(np.transpose(s3),inpDataB)
    #print("w1Err.shape",w1Err.shape)

    #w1Err = w1Err[1:,:]

    # Regularization
    
    # Equation 15: Regularization term 
 
    obj_val = nll + lambdaval * ((np.sum(w1**2) + np.sum(w2**2))/(2*dataLen))
    #print("w1",w1.shape)
    # Equation 16 & 17: Gradient
    #print("lamb",(lambdaval*w1).shape)
    
    w1Grad = (w1Err[1:,:] + lambdaval*w1)/dataLen
    w2Grad = (w2Err + lambdaval*w2)/dataLen

    #print("gradshape",w1Grad.shape,w2Grad.shape)

    


    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    obj_grad = np.concatenate((w1Grad.flatten(),w2Grad.flatten())) 

    #print("grad", obj_grad)
    print("obj_val", obj_val)

    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    
    
    # Equation 1: forward to hidden layer 1
    
    inpData = np.c_[np.ones(data.shape[0]),data] # nrow x ncol+1
    ly1Op=np.dot(inpData,np.transpose(w1))  #nrows x 50
    
    # Equation 2: Activation function for hidden layer 1
    l1Act = sigmoid(ly1Op)
    
    # Equation 3: Propogate to output layer
    l1Act = np.c_[np.ones(l1Act.shape[0]),l1Act] #nrows x 51
    opLy = np.dot(l1Act,np.transpose(w2)) #nrows x 10

    # Equation 4: Activation function for output layer
    op = sigmoid(opLy)

    labels = np.array([])

    labels = np.argmax(op,axis=1)
    #print(labels.shape)
    #labels = labels.reshape(len(labels),1)
    # Your code here

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#neuronsList = [4,8,12,16,20]
#lambdaList = [0,10,20,30,40,50,60]

neuronsList = [20]
lambdaList = [0]

#  Train Neural Networks

# Place holder
accDF = pd.DataFrame(columns = ['Nuerons','Lambda','TrainAcc','ValidAcc','TestAcc','ExeTime'])


for nuerons in neuronsList:
    for lambdaVals in lambdaList:


        # set the number of nodes in input unit (not including bias unit)
        n_input = train_data.shape[1]

        # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = nuerons

        # set the number of nodes in output unit
        n_class = 10

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        # set the regularization hyper-parameter
        lambdaval = lambdaVals

        print("Number of Neurons",nuerons)
        print("Lambda",lambdaVals)

        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

        opts = {'maxiter': 50}  # Preferred value.

        startTime = time.time()
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        endTime = time.time()

        exeTime = (endTime-startTime)

        print("Training Time : ",exeTime)

        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


        # Test the computed parameters

        predicted_label = nnPredict(w1, w2, train_data)

        # find the accuracy on Training Dataset
        print("Shape check",predicted_label.shape,train_label.shape)
        #trainAcc = 100 * np.mean((predicted_label == train_label.reshape(train_label.shape[0],1)))
        trainAcc = 100 * np.mean((predicted_label == train_label))
        print('\n Training set Accuracy:' + str(trainAcc) + '%')

        predicted_label = nnPredict(w1, w2, validation_data)

        # find the accuracy on Validation Dataset
        #validationAcc = 100 * np.mean((predicted_label == validation_label.reshape(validation_label.shape[0],1)))
        validationAcc = 100 * np.mean((predicted_label == validation_label))

        print('\n Validation set Accuracy:' + str(validationAcc) + '%')


        predicted_label = nnPredict(w1, w2, test_data)

        # find the accuracy on Validation Dataset
        #testAcc = 100 * np.mean((predicted_label == test_label.reshape(test_label.shape[0],1)))
        testAcc = 100 * np.mean((predicted_label == test_label))

        print('\n Test set Accuracy:' + str(testAcc) + '%')

        #caputuring data
        exeDF = pd.DataFrame([[nuerons,lambdaVals,trainAcc,validationAcc,testAcc,exeTime]],columns = ['Nuerons','Lambda','TrainAcc','ValidAcc','TestAcc','ExeTime'])

        accDF = accDF.append(exeDF)

print("Accuracy stats")
print(accDF)
