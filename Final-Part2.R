###########################4335 Final##############################
# install following packages if haven't installed before
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("mxnet")
library(caret)
library(mxnet)

################################################
############## Preparation #####################
################################################
# Load the MNIST digit recognition dataset into R
# dataset downloaded from http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# Copyright:2008, Brendan O'Connor - gist.github.com/39760 - anyall.org

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('Minist/train-images-idx3-ubyte')
  test <<- load_image_file('Minist/t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('Minist/train-labels-idx1-ubyte')
  test$y <<- load_label_file('Minist/t10k-labels-idx1-ubyte')  
}

show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

train <- data.frame()
test <- data.frame()

# Load data.
load_mnist()

show_digit(train$x[1000,])

# Setup training data with digit and pixel values with 50/50 split for train/cv.
inTrain = data.frame(y=train$y, train$x)
inTrain$y <- as.factor(inTrain$y)
trainIndex = createDataPartition(inTrain$y, p = 0.067,list=FALSE)

training = inTrain[trainIndex,]
testing = data.frame(test$y, test$x)

train.x = data.matrix(training[,2:785])
train.y = training[,1]



###################################################
################## DNN ############################
###################################################
### download package from the following website
# http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/6/index.html
# warning: running the package need installing Java

### Install H2O package in R
## Following code from dictionary of H2O package
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
if (! ("jsonlite" %in% rownames(installed.packages()))) { install.packages("jsonlite") }
if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }
install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/6/R")))

## load the package 
library("h2o")
localH2O = h2o.init(max_mem_size = '6g',nthreads = -1)

# prepare data 
training[,1] = as.factor(training[,1]) # convert digit labels to factor for classification
train_h2o = as.h2o(training)
test_h2o = as.h2o(testing)



######## try different epochs to test convergence ################
# try epochs: 5,10,15,20......300
training_acc=c()
testing_acc=c()

for(i in seq(5,300,5)){
  ## train model
  dnn.model = h2o.deeplearning(
    x = 2:785,
    y = 1,   # column number for label
    training_frame = train_h2o, # data in H2O format
    hidden = c(100,100), # no. of neurons in each hidden layer
    activation = "RectifierWithDropout", # activation function
    hidden_dropout_ratios = c(0.5,0.5), # percent of neurons dropout
    nesterov_accelerated_gradient = T, # use it for speed
    epochs = i)
  
  ## Predict on testing set & calculate testing accuracy
  pred <- h2o.predict(dnn.model, test_h2o)
  # Accuracy
  train_accuracy <- 1-x$Error[dim(x)[1]]
  test_accuracy <- sum(diag(table(as.vector(test_h2o[, 1]), as.vector(pred[,1]) )))/10000
  
  ## Record outputs in two vectors
  x=h2o.confusionMatrix(dnn.model)
  training_acc[length(training_acc)+1] = train_accuracy
  testing_acc[length(testing_acc)+1] = test_accuracy
  
  ## Print results
  print(paste("epochs = ",i))
  print(paste("training accuracy = ",train_accuracy))
  print(paste("testing accuracy = ",test_accuracy))
}


## Plot learning curve
plot(seq(5,300,5),training_acc,col="blue",xlab="Epochs",ylab="Accuracy",ylim=c(0.89,1))
lines(seq(5,300,5),training_acc,col="blue",xlab="Epochs",ylab="Accuracy")
points(seq(5,300,5),testing_acc,col="red",xlab="Epochs",ylab="Accuracy")
lines(seq(5,300,5),testing_acc,col="red",xlab="Epochs",ylab="Accuracy")

legend("bottomright", legend = c("train accuracy","test accuracy"), col = c("blue","red"),
       lty=c(1,1), lwd=c(2,2),)

grid(col = "lightgray", lty = "dotted",lwd = par("lwd"), equilogs = TRUE)



######## Tesing Various Model Parameters ################
dnn.m1 = h2o.deeplearning(
  x = 2:785,
  y = 1,   # column number for label
  training_frame = train_h2o, # data in H2O format
  hidden = c(500,500), # no. of neurons in each hidden layer
  hidden_dropout_ratios = c(0.5,0.5), # percent of neurons dropout
  nesterov_accelerated_gradient = T, # use it for speedn
  # loss =c("CrossEntropy"), # unccoment if using Logistic activation function
  epsilon = 0.00001, ##learning rate
  l1=1e-5,  ## add some L1/L2 regularization
  l2=1e-5,
  epochs = 100)

plot(dnn.m1)

## training accuracy
x=h2o.confusionMatrix(dnn.m1)
acc=1-x$Error[dim(x)[1]]
acc

## Predict on testing set & calculate testing accuracy
pred <- h2o.predict(dnn.m1, test_h2o)
# Accuracy
sum(diag(table(as.vector(test_h2o[, 1]), as.vector(pred[,1]) )))/10000




######## Final Model ################
dnn.m1 = h2o.deeplearning(
  x = 2:785,
  y = 1,   # column number for label
  training_frame = train_h2o, # data in H2O format
  hidden = c(500,300,100),
  activation = "RectifierWithDropout", # activation function
  input_dropout_ratio = 0.2, # % of inputs dropout
  hidden_dropout_ratios = c(0.5,0.5,0.5), # percent of neurons dropout
  nesterov_accelerated_gradient = T, # use it for speedn
  l1=1e-5,  ## add some L1/L2 regularization
  l2=1e-5,
  epochs = 100)

plot(dnn.m1)

## training accuracy
x=h2o.confusionMatrix(dnn.m1)
acc=1-x$Error[dim(x)[1]]
acc

## Predict on testing set & calculate testing accuracy
pred <- h2o.predict(dnn.m1, test_h2o)
# Accuracy
sum(diag(table(as.vector(test_h2o[, 1]), as.vector(pred[,1]) )))/10000


