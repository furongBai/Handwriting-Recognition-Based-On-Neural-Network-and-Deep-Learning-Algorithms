###########################4335 Final##############################

install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
library(caret)
library(mxnet)

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
  train <<- load_image_file('train-images-idx3-ubyte')
  test <<- load_image_file('t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('train-labels-idx1-ubyte')
  test$y <<- load_label_file('t10k-labels-idx1-ubyte')  
}

show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

train <- data.frame()
test <- data.frame()

# Load data.
load_mnist()

# Normalize: X = (X - min) / (max - min) => X = (X - 0) / (255 - 0) => X = X / 255.
# train$x <- train$x / 255

show_digit(train$x[8828,])

# Setup training data with digit and pixel values with 50/50 split for train/cv.
# with cross-validation
inTrain = data.frame(y=train$y, train$x)
inTrain$y <- as.factor(inTrain$y)

trainIndex = createDataPartition(inTrain$y, p = 0.15,list=FALSE)

training = inTrain[trainIndex,]
cv = inTrain[-trainIndex,]
testing = data.frame(test$y, test$x)

train.x = data.matrix(training[,2:785])
train.y = training[,1]

mx.set.seed(10)

model = mx.mlp(train.x, train.y, hidden_node = 500, out_node = 10,
               out_activation="softmax", num.round=200, 
               learning.rate=0.0008, momentum=0.7,
               eval.metric=mx.metric.accuracy)
proc.time()

# test on testing set
testing.x <- t(testing[, -1])
testing.y <- test$y
preds = predict(model, testing.x)
dim(preds)
pred.label <- max.col(t(preds)) - 1
table(pred.label)

# Accuracy
sum(diag(table(testing[, 1], pred.label)))/10000


