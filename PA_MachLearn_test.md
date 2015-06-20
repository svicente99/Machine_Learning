# Machine Learning: <br><small><b>Peer Assessment - Course Project</b></small>
<a href='mailto:svicente99@yahoo.com' title='twitter:@svicente99'>Sergio Vicente</a>  
May, 2015  

<br><br>
You may get source code of this at <https://github.com/svicente99/MachLearn_PeerAssessment> 

And html version is available at RPubs: <http://rpubs.com/svicente99/MachLearn_Peer_Assesment>

* * * *

#### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways [A,B,C,D,E].

#### Objective and development

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. Any of the other variables were used to predict it. Along next sections we describe how the model was built and fitted. Two alternatives were tested: one with classification tree and another one with random forest The expected out of sample error was showed in order to advice the choice we did. Our option was **random forest** as best model fit. Finally we use this model to predict 20 different test cases. 

---------

#### Data Sourcing

The data for this assignment can be downloaded from the course web site:

* Dataset: <a href="http://groupware.les.inf.puc-rio.br/har.">Activity Data</a> [12.2 MBytes]

There is also some documentation of the database available. See above links to know how some of the variables are constructed/defined.

---------

#### Data Accquisiton

Setting parameters and files to be processed:


```r
# MAIN PARAMETERS
DATA_FOLDER <- "./data"  # subdirectory of current named 'data'
URL_DATA <- "http://d396qusza40orc.cloudfront.net/predmachlearn"
CSV_TRAIN <- "pml-training.csv"
CSV_TEST <- "pml-testing.csv"
# -----------------------------------------------------------------
```

Getting CSV data files and setting the main data frame (df_*):


```
## [1] "Dimension of training data - rows x cols"
```

```
## [1] 19622   160
```

```
## [1] "Dimension of test data - rows x cols"
```

```
## [1]  20 160
```

#### Packages loading

The next Libraries were used in this project. A custom function will install them (if not done yet) and will load on your "RStudio" environment.




```r
# we just install and load the libraries we need to do this analysis
pkgTest("rpart")
```

```
## Loading required package: rpart
```

```r
pkgTest("rpart.plot")
```

```
## Loading required package: rpart.plot
```

```r
pkgTest("caret")
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pkgTest("rattle")
```

```
## Loading required package: rattle
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
pkgTest("randomForest")
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(rpart)
library(rpart.plot)
library(caret)
library(rattle)
library(randomForest)
```

#### Data cleaning

We need to pass a good filter to clean these data up. Next function do this job.
It's not interesting to include in this analysis data containing missing values. So, get rid of them:


```r
clean_data <- function(df, step3=TRUE) 
{
  # get rid of columns that has only "missing values"
	df_clean <- df[, colSums(is.na(df)) == 0] 
	print(dim(df_clean))
	# then throw columns out that not participate as predictors
    vColsNotPred <- c("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
    cols2Remove <- which(colnames(df) %in% vColsNotPred)
    df_clean <- df_clean[, -cols2Remove]	
	print(dim(df_clean))
  # the ID column (the 1st) is another that not influences our forecasting
  df_clean <- df_clean[c(-1)]
	# lastly, remove columns with zero variability
  if(step3) { ## because there is no need to 'testing' dataset ##
  	colsNear0 <- nearZeroVar(df_clean)
  	df_clean <- df_clean[, -colsNear0]
  	print(dim(df_clean))
  }
  return(df_clean)
}

df_train <- clean_data(df_train) 
```

```
## [1] 19622    93
## [1] 19622    87
## [1] 19622    53
```

```r
summary(df_train$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

#### Spliting data in two subsets - train & test

We'll split training data set, after cleaning, in two subsets - one is a pure training data set, containing **70%** of observations and another, smaller (30%) to contain data for validation.


```r
set.seed(1987)      ## to may reproducible research
trainPart <- createDataPartition(df_train$classe, p=0.7, list=FALSE)
training <- df_train[+trainPart,]
testing  <- df_train[-trainPart,]
rm(trainPart)

dim(training); dim(testing)
```

```
## [1] 13737    53
```

```
## [1] 5885   53
```

```r
class(training)
```

```
## [1] "data.frame"
```

```r
training$classe <- as.factor(training$classe)

# Equalizing dimensions (columns) to test data sets

nCols <- ncol(training)
colsTraining1 <- colnames(training)
colsTraining2 <- colnames(training[, -nCols])  # remove classe column (for test data set)
testing <- testing[colsTraining1]           # leave vars that are also in training set
df_test <- df_test[colsTraining2]           # leave vars that are also in training set

summary(training$classe); summary(testing$classe)
```

```
##    A    B    C    D    E 
## 3906 2658 2396 2252 2525
```

```
##    A    B    C    D    E 
## 1674 1139 1026  964 1082
```

#### Model Fitting

We initially use 'decision tree' technique to fit a model in these data. Here are the results obtained to this adjusted model and its tree plotting.


```
## 
## Classification tree:
## rpart(formula = classe ~ ., data = training, method = "class")
## 
## Variables actually used in tree construction:
##  [1] accel_dumbbell_y     accel_dumbbell_z     accel_forearm_x     
##  [4] magnet_arm_y         magnet_belt_z        magnet_dumbbell_y   
##  [7] magnet_dumbbell_z    magnet_forearm_z     pitch_belt          
## [10] pitch_forearm        roll_belt            roll_dumbbell       
## [13] roll_forearm         total_accel_dumbbell yaw_belt            
## 
## Root node error: 9831/13737 = 0.71566
## 
## n= 13737 
## 
##          CP nsplit rel error  xerror      xstd
## 1  0.115553      0   1.00000 1.00000 0.0053780
## 2  0.059336      1   0.88445 0.88445 0.0057464
## 3  0.035398      4   0.70644 0.70735 0.0059605
## 4  0.031431      5   0.67104 0.66982 0.0059559
## 5  0.022124      6   0.63961 0.64022 0.0059401
## 6  0.019835     11   0.51439 0.51490 0.0057511
## 7  0.018716     12   0.49456 0.49456 0.0057010
## 8  0.018513     13   0.47584 0.47289 0.0056412
## 9  0.014749     15   0.43882 0.44217 0.0055448
## 10 0.013122     16   0.42407 0.42498 0.0054846
## 11 0.011189     18   0.39782 0.41267 0.0054387
## 12 0.011087     19   0.38663 0.39630 0.0053738
## 13 0.010000     20   0.37555 0.38409 0.0053226
```

![](PA_MachLearn_test_files/figure-html/unnamed-chunk-7-1.png) 

For next, we use 'random forest' algorithm to fit another model to those data. Follow the results.


```r
modelFit2 <- randomForest(classe ~. , data=training)
modelFit2
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.5%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    2    0    0    0 0.0005120328
## B   13 2640    5    0    0 0.0067720090
## C    0    6 2388    2    0 0.0033388982
## D    0    0   25 2224    3 0.0124333925
## E    0    0    1   11 2513 0.0047524752
```

#### Predictions building and comparison


```r
prev1 <- predict(modelFit1, data=testing, type="class")
prev2 <- predict(modelFit2, data=testing, type="class")

tab <- table(prev1, prev2)
apply( 
  prop.table(tab,1)*100, 2, 
    function(u) sprintf( "%.1f%%", u ) 
)
```

```
##       prev2
##        A       B       C       D       E      
##   [1,] "74.9%" "11.4%" "3.5%"  "7.7%"  "2.5%" 
##   [2,] "4.3%"  "74.3%" "9.5%"  "4.0%"  "7.9%" 
##   [3,] "3.3%"  "6.9%"  "66.9%" "12.0%" "10.8%"
##   [4,] "4.4%"  "10.0%" "9.3%"  "70.9%" "5.4%" 
##   [5,] "2.9%"  "5.8%"  "2.7%"  "9.6%"  "79.1%"
```

### Using prediction model on Test data provided


```r
myChoiceModel <- predict(modelFit2, df_test, type="class")
myChoiceModel
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
# Write the result file to submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./data/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(myChoiceModel)
# Twenty files were saved on data folder (check it!).
```
