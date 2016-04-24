# Machine Learning
# WK4 Project
# Tom McMurtrie

library(downloader)
library(caret)
library(dplyr)
library(rpart)
library(randomForest)

# Download the data.
trainfileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testfileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if(!file.exists("pml-training.csv")) {
    download(trainfileUrl, dest = "pml-training.csv", mode = "wb")
}

if(!file.exists("pml-testing.csv")) {
    download(testfileUrl, dest = "pml-testing.csv", mode = "wb")
}


# Read in the data.
training <- read.csv("pml-training.csv")
dim(training)


# take out variables w/ near zero variance
nzv_train <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[ , nzv_train$nzv == "FALSE"] #renamed working dataset to training2 to preserve integrity of training set

# take out variables w/ NAs
training <- training[ , colSums(is.na(training)) == 0]

# take out "X" factor and time information variables
training <- select(training, -c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
tng_colnames <- select(training, -c(classe))


# Repeat above steps for testing data
testing <- read.csv("pml-testing.csv")
dim(testing)
#testing$classe <- "A"
testing <- testing[as.character(colnames(tng_colnames))]
rm(tng_colnames)


# Subset the training set for cross validation
set.seed(125) #changed from 125
inTrain <- createDataPartition(y = training$classe, p = 0.65, list = FALSE) #maybe lower to 65% or so
train_set <- training[inTrain, ]
val <- training[-inTrain, ]

## put in a section for rpart examination ##
mod_tree <- train(classe ~ ., data = train_set, method = "rpart")
mod_tree
pred_val_tree <- predict(mod_tree, val)
confusionMatrix(val$classe, pred_val_tree) #a coin flip of predictive power


## worked but took a while ##
#RF_controls <- trainControl(method = "cv", 5)
modFit <- train(classe ~ ., data = train_set, method = "rf", prox = TRUE, trControl = trainControl(method = "cv", number = 10))
#modFit <- train(classe ~ ., data = train_set, method = "rf", prox = TRUE, trainControl = RF_controls)
modFit
modFit$finalModel 

# Predict against cross-val set
pred_val <- predict(modFit, val)
confusionMatrix(val$classe, pred_val)

# Prediction of the test variable
pred_test <- predict(modFit, testing)
pred_test
