# Load the MNIST digit recognition dataset into R
# dataset downloaded from http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# Copyright:2008, Brendan O'Connor - gist.github.com/39760 - anyall.org

# download mxnet package from github
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
require(mxnet)
library(mxnet)

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
  train <<- load_image_file('train-images-idx3-ubyte')
  test <<- load_image_file('t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('train-labels-idx1-ubyte')
  test$y <<- load_label_file('t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

# Load data.
train <- data.frame()
test <- data.frame()
load_mnist()

library(lattice)
library(ggplot2)
library(caret)

inTrain = data.frame(y=train$y, train$x)
inTrain$y <- as.factor(inTrain$y)
trainIndex = createDataPartition(inTrain$y, p = 0.20,list=FALSE)
training = inTrain[trainIndex,]

training.x = t(training[, -1]) 
training.y = training[, 1]     	
table(training.y)


train.array <- training.x
dim(train.array) <- c(28, 28, 1, ncol(training.x))


testing = data.frame(y=train$y,test$x)
testing.x = t(testing[, -1]) 
test.array <- testing.x
dim(test.array) <- c(28, 28, 1, ncol(testing.x))



# CNN Configuration

# input
data <- mx.symbol.Variable('data')
# first convolution
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second convolution
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first full conneted layer
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second full connected layer
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
#Loss function: Softmax
cnn <- mx.symbol.SoftmaxOutput(data=fc2,name = "cnn")


devices <- mx.cpu()
mx.set.seed(0)
# train the model
cnn.model <- mx.model.FeedForward.create(cnn, X=train.array, 
  y=training.y, ctx=devices, num.round=30, 
  array.batch.size=100, learning.rate=0.06, 
  momentum=0.7, eval.metric=mx.metric.accuracy,
  initializer=mx.init.uniform(0.05),
  epoch.end.callback=mx.callback.log.train.metric(100)
  )

# prediction
preds <- predict(cnn.model, test.array)
pred.label <- max.col(t(preds)) - 1
table(pred.label)
sum(diag(table(testing[,1],pred.label)))/10000
