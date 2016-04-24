# Predicting How Well Participants Perform Training Activities
Thomas McMurtrie  
April 23, 2016  

##Overview
People regularly quantify how much of a particular activity they do, but they rarely quantify how well they do it.  In this project, I attempt to use data from accelerometers on the belt, forearm, army, and dumbell of 6 participants in order to develop a means to predict correct activity execution.  

##Background
Using devides such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activity relatively inexpensively.  THese types of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.  

One thing people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  In this project, I will use data from the accelerometers on the belt, forearm, arm, and dumbell of six participants.  THey were asked to perform barbell lifts correctly and incorrectly in five different ways.  More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har).  

The  training data used to build the initial model is found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv); the data used for testing found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).  

##Data Processing
###Loading the Data
To begin the process of answering the research question, I first download the datasets.  Because the datasets are large, the code first ensures that the data files have not been previously downloaded.   

```r
library(downloader)
```

```
## Warning: package 'downloader' was built under R version 3.2.3
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.4
```

```
## Loading required package: lattice
```

```
## Warning: package 'lattice' was built under R version 3.2.3
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.3
```

```r
library(dplyr)
```

```
## Warning: package 'dplyr' was built under R version 3.2.3
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.2.3
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.4
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
# Download the datasets.
trainfileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testfileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if(!file.exists("pml-training.csv")) {
    download(trainfileUrl, dest = "pml-training.csv", mode = "wb")
}

if(!file.exists("pml-testing.csv")) {
    download(testfileUrl, dest = "pml-testing.csv", mode = "wb")
}
```

After downloading the files, I then read the data into R--beginning with the training dataset.  

```r
# Read in the data.
training <- read.csv("pml-training.csv")
dim(training)
```

```
## [1] 19622   160
```
I can see that the training dataset has 160 variables and 19,622 observations.  

###Processing the Data
I first apply the nearZeroVar function in order to remove those variables contained in the training set that have near zero variability.  This helps to reduce the dimensions of the dataset and eliminate variables that will not provide predictive power to the modeling effort.

```r
nzv_train <- nearZeroVar(training, saveMetrics = TRUE)
nzv_train
```

```
##                            freqRatio percentUnique zeroVar   nzv
## X                           1.000000  100.00000000   FALSE FALSE
## user_name                   1.100679    0.03057792   FALSE FALSE
## raw_timestamp_part_1        1.000000    4.26562022   FALSE FALSE
## raw_timestamp_part_2        1.000000   85.53154622   FALSE FALSE
## cvtd_timestamp              1.000668    0.10192641   FALSE FALSE
## new_window                 47.330049    0.01019264   FALSE  TRUE
## num_window                  1.000000    4.37264295   FALSE FALSE
## roll_belt                   1.101904    6.77810621   FALSE FALSE
## pitch_belt                  1.036082    9.37722964   FALSE FALSE
## yaw_belt                    1.058480    9.97349913   FALSE FALSE
## total_accel_belt            1.063160    0.14779329   FALSE FALSE
## kurtosis_roll_belt       1921.600000    2.02323922   FALSE  TRUE
## kurtosis_picth_belt       600.500000    1.61553358   FALSE  TRUE
## kurtosis_yaw_belt          47.330049    0.01019264   FALSE  TRUE
## skewness_roll_belt       2135.111111    2.01304658   FALSE  TRUE
## skewness_roll_belt.1      600.500000    1.72255631   FALSE  TRUE
## skewness_yaw_belt          47.330049    0.01019264   FALSE  TRUE
## max_roll_belt               1.000000    0.99378249   FALSE FALSE
## max_picth_belt              1.538462    0.11211905   FALSE FALSE
## max_yaw_belt              640.533333    0.34654979   FALSE  TRUE
## min_roll_belt               1.000000    0.93772296   FALSE FALSE
## min_pitch_belt              2.192308    0.08154113   FALSE FALSE
## min_yaw_belt              640.533333    0.34654979   FALSE  TRUE
## amplitude_roll_belt         1.290323    0.75425543   FALSE FALSE
## amplitude_pitch_belt        3.042254    0.06625217   FALSE FALSE
## amplitude_yaw_belt         50.041667    0.02038528   FALSE  TRUE
## var_total_accel_belt        1.426829    0.33126083   FALSE FALSE
## avg_roll_belt               1.066667    0.97339721   FALSE FALSE
## stddev_roll_belt            1.039216    0.35164611   FALSE FALSE
## var_roll_belt               1.615385    0.48924676   FALSE FALSE
## avg_pitch_belt              1.375000    1.09061258   FALSE FALSE
## stddev_pitch_belt           1.161290    0.21914178   FALSE FALSE
## var_pitch_belt              1.307692    0.32106819   FALSE FALSE
## avg_yaw_belt                1.200000    1.22311691   FALSE FALSE
## stddev_yaw_belt             1.693878    0.29558659   FALSE FALSE
## var_yaw_belt                1.500000    0.73896647   FALSE FALSE
## gyros_belt_x                1.058651    0.71348486   FALSE FALSE
## gyros_belt_y                1.144000    0.35164611   FALSE FALSE
## gyros_belt_z                1.066214    0.86127816   FALSE FALSE
## accel_belt_x                1.055412    0.83579655   FALSE FALSE
## accel_belt_y                1.113725    0.72877383   FALSE FALSE
## accel_belt_z                1.078767    1.52379982   FALSE FALSE
## magnet_belt_x               1.090141    1.66649679   FALSE FALSE
## magnet_belt_y               1.099688    1.51870350   FALSE FALSE
## magnet_belt_z               1.006369    2.32901845   FALSE FALSE
## roll_arm                   52.338462   13.52563449   FALSE FALSE
## pitch_arm                  87.256410   15.73234125   FALSE FALSE
## yaw_arm                    33.029126   14.65701763   FALSE FALSE
## total_accel_arm             1.024526    0.33635715   FALSE FALSE
## var_accel_arm               5.500000    2.01304658   FALSE FALSE
## avg_roll_arm               77.000000    1.68178575   FALSE  TRUE
## stddev_roll_arm            77.000000    1.68178575   FALSE  TRUE
## var_roll_arm               77.000000    1.68178575   FALSE  TRUE
## avg_pitch_arm              77.000000    1.68178575   FALSE  TRUE
## stddev_pitch_arm           77.000000    1.68178575   FALSE  TRUE
## var_pitch_arm              77.000000    1.68178575   FALSE  TRUE
## avg_yaw_arm                77.000000    1.68178575   FALSE  TRUE
## stddev_yaw_arm             80.000000    1.66649679   FALSE  TRUE
## var_yaw_arm                80.000000    1.66649679   FALSE  TRUE
## gyros_arm_x                 1.015504    3.27693405   FALSE FALSE
## gyros_arm_y                 1.454369    1.91621649   FALSE FALSE
## gyros_arm_z                 1.110687    1.26388747   FALSE FALSE
## accel_arm_x                 1.017341    3.95984099   FALSE FALSE
## accel_arm_y                 1.140187    2.73672409   FALSE FALSE
## accel_arm_z                 1.128000    4.03628580   FALSE FALSE
## magnet_arm_x                1.000000    6.82397309   FALSE FALSE
## magnet_arm_y                1.056818    4.44399144   FALSE FALSE
## magnet_arm_z                1.036364    6.44684538   FALSE FALSE
## kurtosis_roll_arm         246.358974    1.68178575   FALSE  TRUE
## kurtosis_picth_arm        240.200000    1.67159311   FALSE  TRUE
## kurtosis_yaw_arm         1746.909091    2.01304658   FALSE  TRUE
## skewness_roll_arm         249.558442    1.68688207   FALSE  TRUE
## skewness_pitch_arm        240.200000    1.67159311   FALSE  TRUE
## skewness_yaw_arm         1746.909091    2.01304658   FALSE  TRUE
## max_roll_arm               25.666667    1.47793293   FALSE  TRUE
## max_picth_arm              12.833333    1.34033228   FALSE FALSE
## max_yaw_arm                 1.227273    0.25991234   FALSE FALSE
## min_roll_arm               19.250000    1.41677709   FALSE  TRUE
## min_pitch_arm              19.250000    1.47793293   FALSE  TRUE
## min_yaw_arm                 1.000000    0.19366018   FALSE FALSE
## amplitude_roll_arm         25.666667    1.55947406   FALSE  TRUE
## amplitude_pitch_arm        20.000000    1.49831821   FALSE  TRUE
## amplitude_yaw_arm           1.037037    0.25991234   FALSE FALSE
## roll_dumbbell               1.022388   84.20650290   FALSE FALSE
## pitch_dumbbell              2.277372   81.74498012   FALSE FALSE
## yaw_dumbbell                1.132231   83.48282540   FALSE FALSE
## kurtosis_roll_dumbbell   3843.200000    2.02833554   FALSE  TRUE
## kurtosis_picth_dumbbell  9608.000000    2.04362450   FALSE  TRUE
## kurtosis_yaw_dumbbell      47.330049    0.01019264   FALSE  TRUE
## skewness_roll_dumbbell   4804.000000    2.04362450   FALSE  TRUE
## skewness_pitch_dumbbell  9608.000000    2.04872082   FALSE  TRUE
## skewness_yaw_dumbbell      47.330049    0.01019264   FALSE  TRUE
## max_roll_dumbbell           1.000000    1.72255631   FALSE FALSE
## max_picth_dumbbell          1.333333    1.72765263   FALSE FALSE
## max_yaw_dumbbell          960.800000    0.37203139   FALSE  TRUE
## min_roll_dumbbell           1.000000    1.69197839   FALSE FALSE
## min_pitch_dumbbell          1.666667    1.81429008   FALSE FALSE
## min_yaw_dumbbell          960.800000    0.37203139   FALSE  TRUE
## amplitude_roll_dumbbell     8.000000    1.97227602   FALSE FALSE
## amplitude_pitch_dumbbell    8.000000    1.95189073   FALSE FALSE
## amplitude_yaw_dumbbell     47.920200    0.01528896   FALSE  TRUE
## total_accel_dumbbell        1.072634    0.21914178   FALSE FALSE
## var_accel_dumbbell          6.000000    1.95698706   FALSE FALSE
## avg_roll_dumbbell           1.000000    2.02323922   FALSE FALSE
## stddev_roll_dumbbell       16.000000    1.99266130   FALSE FALSE
## var_roll_dumbbell          16.000000    1.99266130   FALSE FALSE
## avg_pitch_dumbbell          1.000000    2.02323922   FALSE FALSE
## stddev_pitch_dumbbell      16.000000    1.99266130   FALSE FALSE
## var_pitch_dumbbell         16.000000    1.99266130   FALSE FALSE
## avg_yaw_dumbbell            1.000000    2.02323922   FALSE FALSE
## stddev_yaw_dumbbell        16.000000    1.99266130   FALSE FALSE
## var_yaw_dumbbell           16.000000    1.99266130   FALSE FALSE
## gyros_dumbbell_x            1.003268    1.22821323   FALSE FALSE
## gyros_dumbbell_y            1.264957    1.41677709   FALSE FALSE
## gyros_dumbbell_z            1.060100    1.04984201   FALSE FALSE
## accel_dumbbell_x            1.018018    2.16593619   FALSE FALSE
## accel_dumbbell_y            1.053061    2.37488533   FALSE FALSE
## accel_dumbbell_z            1.133333    2.08949139   FALSE FALSE
## magnet_dumbbell_x           1.098266    5.74864948   FALSE FALSE
## magnet_dumbbell_y           1.197740    4.30129447   FALSE FALSE
## magnet_dumbbell_z           1.020833    3.44511263   FALSE FALSE
## roll_forearm               11.589286   11.08959331   FALSE FALSE
## pitch_forearm              65.983051   14.85577413   FALSE FALSE
## yaw_forearm                15.322835   10.14677403   FALSE FALSE
## kurtosis_roll_forearm     228.761905    1.64101519   FALSE  TRUE
## kurtosis_picth_forearm    226.070588    1.64611151   FALSE  TRUE
## kurtosis_yaw_forearm       47.330049    0.01019264   FALSE  TRUE
## skewness_roll_forearm     231.518072    1.64611151   FALSE  TRUE
## skewness_pitch_forearm    226.070588    1.62572623   FALSE  TRUE
## skewness_yaw_forearm       47.330049    0.01019264   FALSE  TRUE
## max_roll_forearm           27.666667    1.38110284   FALSE  TRUE
## max_picth_forearm           2.964286    0.78992967   FALSE FALSE
## max_yaw_forearm           228.761905    0.22933442   FALSE  TRUE
## min_roll_forearm           27.666667    1.37091020   FALSE  TRUE
## min_pitch_forearm           2.862069    0.87147080   FALSE FALSE
## min_yaw_forearm           228.761905    0.22933442   FALSE  TRUE
## amplitude_roll_forearm     20.750000    1.49322189   FALSE  TRUE
## amplitude_pitch_forearm     3.269231    0.93262664   FALSE FALSE
## amplitude_yaw_forearm      59.677019    0.01528896   FALSE  TRUE
## total_accel_forearm         1.128928    0.35674243   FALSE FALSE
## var_accel_forearm           3.500000    2.03343186   FALSE FALSE
## avg_roll_forearm           27.666667    1.64101519   FALSE  TRUE
## stddev_roll_forearm        87.000000    1.63082255   FALSE  TRUE
## var_roll_forearm           87.000000    1.63082255   FALSE  TRUE
## avg_pitch_forearm          83.000000    1.65120783   FALSE  TRUE
## stddev_pitch_forearm       41.500000    1.64611151   FALSE  TRUE
## var_pitch_forearm          83.000000    1.65120783   FALSE  TRUE
## avg_yaw_forearm            83.000000    1.65120783   FALSE  TRUE
## stddev_yaw_forearm         85.000000    1.64101519   FALSE  TRUE
## var_yaw_forearm            85.000000    1.64101519   FALSE  TRUE
## gyros_forearm_x             1.059273    1.51870350   FALSE FALSE
## gyros_forearm_y             1.036554    3.77637346   FALSE FALSE
## gyros_forearm_z             1.122917    1.56457038   FALSE FALSE
## accel_forearm_x             1.126437    4.04647844   FALSE FALSE
## accel_forearm_y             1.059406    5.11160942   FALSE FALSE
## accel_forearm_z             1.006250    2.95586586   FALSE FALSE
## magnet_forearm_x            1.012346    7.76679238   FALSE FALSE
## magnet_forearm_y            1.246914    9.54031189   FALSE FALSE
## magnet_forearm_z            1.000000    8.57710733   FALSE FALSE
## classe                      1.469581    0.02548160   FALSE FALSE
```

```r
training <- training[ , nzv_train$nzv == "FALSE"]
```

Next, I take out the remaining variables that contain "NA" variables.  Variables with significant "NAs" cause issues with many of the predicitive algorithms.

```r
training <- training[ , colSums(is.na(training)) == 0]
```

I remove the "X" variables (just an index variable), and the date/time variables since their does not seem to be a time component to this data.  The tng_colnames object is created as the final listing of variables that I will keep in the "testing" set.  

```r
training <- select(training, -c(X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
tng_colnames <- select(training, -c(classe))
dim(training)
```

```
## [1] 19622    55
```

Next, I read in the testing dataset.  I then subset it to keep only those variables remaining in the training dataset (less the classe variable, of course).  

```r
testing <- read.csv("pml-testing.csv")
dim(testing)
```

```
## [1]  20 160
```

```r
testing <- testing[as.character(colnames(tng_colnames))]
rm(tng_colnames)
```

Finally, because I want to validate my model prior to its application, I subset the training set into two subsets:  the true training set and the portion of the training set I'll use to validate my prediction algorithm.  

```r
set.seed(125)
inTrain <- createDataPartition(y = training$classe, p = 0.65, list = FALSE) #maybe lower to 65% or so
train_set <- training[inTrain, ]
val <- training[-inTrain, ]
```

###Building the Models
I begin my modeling by implementing a simple regression tree.  Given their relative simplicity, I first seek to apply the method before resorting to more computationally intensive methods (like a random forest).  

```r
mod_tree <- train(classe ~ ., data = train_set, method = "rpart")
mod_tree
```

```
## CART 
## 
## 12757 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 12757, 12757, 12757, 12757, 12757, 12757, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03855422  0.5326530  0.40166503
##   0.05991238  0.4215803  0.21950998
##   0.11686747  0.3327038  0.07553272
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03855422.
```

```r
pred_val_tree <- predict(mod_tree, val)
confusionMatrix(val$classe, pred_val_tree)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1778   30  142    0    3
##          B  547  454  327    0    0
##          C  554   41  602    0    0
##          D  522  194  409    0    0
##          E  176  179  354    0  553
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4934          
##                  95% CI : (0.4815, 0.5053)
##     No Information Rate : 0.521           
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3377          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4971  0.50557  0.32824       NA  0.99460
## Specificity            0.9468  0.85353  0.88173   0.8361  0.88762
## Pos Pred Value         0.9104  0.34187  0.50292       NA  0.43819
## Neg Pred Value         0.6338  0.91981  0.78264       NA  0.99946
## Prevalence             0.5210  0.13081  0.26715   0.0000  0.08099
## Detection Rate         0.2590  0.06613  0.08769   0.0000  0.08055
## Detection Prevalence   0.2845  0.19345  0.17436   0.1639  0.18383
## Balanced Accuracy      0.7219  0.67955  0.60499       NA  0.94111
```
This model's roughly 50% accuracy indicates that it is no better than a coin flip.    

I next implement a random forest predictive algorithm against the training dataset.  In my implementation, I accept the (unstated) default of examining 500 trees.  I specify that my model should be controlled (trControl) using the trainControl function that applies a 10 fold cross validation in order to maximize model fit (without overfitting).  The fairly large number of trees and folds resulted in this model taking a lengthy time to process.  

```r
modFit <- train(classe ~ ., data = train_set, method = "rf", prox = TRUE, trControl = trainControl(method = "cv", number = 10))
modFit
```

```
## Random Forest 
## 
## 12757 samples
##    54 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 11482, 11483, 11481, 11481, 11482, 11483, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9927097  0.9907785
##   30    0.9973349  0.9966289
##   58    0.9952186  0.9939519
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 30.
```

```r
modFit$finalModel 
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 30
## 
##         OOB estimate of  error rate: 0.21%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3625    1    0    0    1 0.0005514199
## B    5 2463    1    0    0 0.0024301337
## C    0    6 2219    0    0 0.0026966292
## D    0    0    7 2084    0 0.0033476805
## E    0    1    0    5 2339 0.0025586354
```

From the above output, we can see that the random forest selected a 30-variable model with an accuracy of 0.998 and a Kappa of 0.997.  The out of box estimate of error rate is 0.2%.  

Given these model fit characteristics, I next use the model to predict agains the validation dataset and compare the prediction against the actual values using the confusionMatrix function.

```r
pred_val <- predict(modFit, val)
confusionMatrix(val$classe, pred_val)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1953    0    0    0    0
##          B    1 1321    6    0    0
##          C    0    1 1196    0    0
##          D    0    0    5 1119    1
##          E    0    0    0    1 1261
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9964, 0.9988)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9972          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9995   0.9992   0.9909   0.9991   0.9992
## Specificity            1.0000   0.9987   0.9998   0.9990   0.9998
## Pos Pred Value         1.0000   0.9947   0.9992   0.9947   0.9992
## Neg Pred Value         0.9998   0.9998   0.9981   0.9998   0.9998
## Prevalence             0.2846   0.1926   0.1758   0.1631   0.1838
## Detection Rate         0.2845   0.1924   0.1742   0.1630   0.1837
## Detection Prevalence   0.2845   0.1934   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9997   0.9990   0.9954   0.9990   0.9995
```
Output from the predictions and confusionMatrix indicate that the random forest model built is a good predictor of the classe variable.  

##Predict Output
Finally, I use the random forest model to predict the class variable of the test dataset.

```r
pred_test <- predict(modFit, testing)
pred_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
These predictions will be input as my Week 4 Quiz.

##Conclusion
In order to develop an accurate prediciton model of the classe variable, I was unable to apply a "simple" and computationally un-demanding regression tree algorithm.  Though the random forest algorithm implemented was significantly more computationally time consuming, the predictive power of the model far exceeded the coin flip prediction of the regression tree.  
