# Creditcard_R


Let's go through the provided R code step by step to understand each part.

# Loading Libraries and Data

```
library(readr)
creditcard <- read_csv("Postgres/creditcard.csv")
View(creditcard)
library(readr):
This loads the readr package, which provides functions for reading data, such as CSV files, into R efficiently.
```

- creditcard <- read_csv("Postgres/creditcard.csv"): This line reads the CSV file located at "Postgres/creditcard.csv" into a data frame called creditcard using the read_csv function from the readr package.

- View(creditcard): This opens a spreadsheet-style viewer to inspect the contents of the creditcard data frame.


# Importing Required Libraries and Additional Datasets

```
# Detecting Credit Card Fraud
# Importing Datasets
library(ranger)
library(caret)
library(data.table)
```

- library(ranger): Loads the ranger package, which is used for building fast and scalable random forest models.

- library(caret): Loads the caret package, which provides functions to streamline model training and evaluation.

- library(data.table): Loads the data.table package for fast data manipulation.



# Data Exploration

```
# Data Exploration
dim(creditcard)
head(creditcard,6)
tail(creditcard,6)
table(creditcard$Class)
summary(creditcard$Amount)
names(creditcard)
var(creditcard$Amount)
sd(creditcard$Amount)
```

- dim(creditcard): Returns the dimensions (number of rows and columns) of the creditcard data frame.

- head(creditcard,6): Displays the first six rows of the creditcard data frame.

- tail(creditcard,6): Displays the last six rows of the creditcard data frame.

- table(creditcard$Class): Provides a frequency table of the Class variable, showing the number of occurrences of each class (e.g., fraud vs. non-fraud).

- summary(creditcard$Amount): Provides summary statistics (min, max, mean, median, etc.) for the Amount column.

- names(creditcard): Returns the names of the columns in the creditcard data frame.

- var(creditcard$Amount): Calculates the variance of the Amount column.

- sd(creditcard$Amount): Calculates the standard deviation of the Amount column.


# Data Manipulation

```
# Data Manipulation
head(creditcard)
creditcard$Amount = scale(creditcard$Amount)
NewData = creditcard[,-c(1)]
head(NewData)
View(NewData)
```

- head(creditcard): Displays the first six rows of the creditcard data frame, again for checking.

- creditcard$Amount = scale(creditcard$Amount): Scales (standardizes) the Amount column, centering it around the mean with a standard deviation of 1.

- NewData = creditcard[,-c(1)]: Creates a new data frame NewData that excludes the first column of creditcard (typically an identifier or timestamp column).

- head(NewData): Displays the first six rows of the NewData data frame.

- View(NewData): Opens a spreadsheet-style viewer to inspect the contents of NewData.



# Data Modelling

```
# Data Modelling
library(caTools)
set.seed(123)
data_sample = sample.split(NewData$Class, SplitRatio = 0.80)
train_data = subset(NewData, data_sample == TRUE)
test_data = subset(NewData, data_sample == FALSE)
dim(train_data)
dim(test_data)
```

- library(caTools): Loads the caTools package, which provides functions for data sampling and splitting.

- set.seed(123): Sets the random seed to 123 for reproducibility of the results.

- data_sample = sample.split(NewData$Class, SplitRatio = 0.80): Splits the data into training and testing sets based on the Class column, with 80% of the data allocated for training.

- train_data = subset(NewData, data_sample == TRUE): Creates the training dataset containing 80% of the data.

- test_data = subset(NewData, data_sample == FALSE): Creates the test dataset containing the remaining 20% of the data.

- dim(train_data): Returns the dimensions of the training dataset.

- dim(test_data): Returns the dimensions of the testing dataset.



# Fitting a Logistic Regression Model

```
# Fitting Logistic Regression Model
Logistic_Model = glm(Class ~ ., test_data, family = binomial())
summary(Logistic_Model)
# Visualizing summarized model through the following plots
plot(Logistic_Model)
# ROC Curve to assess the performance of the model
library(pROC)
```

- Logistic_Model = glm(Class ~ ., test_data, family = binomial()): Fits a logistic regression model using all features in test_data to predict Class. The family = binomial() option specifies a logistic regression model.

- summary(Logistic_Model): Displays a summary of the logistic regression model, including coefficients, standard errors, and significance levels.

- plot(Logistic_Model): Generates diagnostic plots for the logistic regression model.

- library(pROC): Loads the pROC package, which is used for analyzing and visualizing ROC curves.



# ROC Curve and Prediction

```
lr.predict <- predict(Logistic_Model, test_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")
```

- lr.predict <- predict(Logistic_Model, test_data, probability = TRUE): Uses the logistic regression model to predict the probabilities of the Class in the test dataset.

- auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue"): Computes and plots the ROC curve for the logistic regression model's predictions, and colors the ROC curve blue. The roc function returns an object that includes the area under the curve (AUC).



# Fitting a Decision Tree Model


```
# Fitting a Decision Tree Model
library(rpart)
library(rpart.plot)
decisionTree_model <- rpart(Class ~ ., creditcard, method = 'class')
predicted_val <- predict(decisionTree_model, creditcard, type = 'class')
probability <- predict(decisionTree_model, creditcard, type = 'prob')
rpart.plot(decisionTree_model)
```

- library(rpart): Loads the rpart package, which is used for creating recursive partitioning and regression trees.

- library(rpart.plot): Loads the rpart.plot package, used for visualizing decision trees.

- decisionTree_model <- rpart(Class ~ ., creditcard, method = 'class'): Trains a decision tree model using all features in the creditcard data frame to predict Class.

- predicted_val <- predict(decisionTree_model, creditcard, type = 'class'): Uses the decision tree model to predict the class labels for the creditcard data.

- probability <- predict(decisionTree_model, creditcard, type = 'prob'): Uses the decision tree model to predict class probabilities for the creditcard data.

- rpart.plot(decisionTree_model): Plots the decision tree structure.



# Fitting an Artificial Neural Network Model


```
# Artificial Neural Network
library(neuralnet)
ANN_model = neuralnet(Class ~ ., train_data, linear.output = FALSE)
plot(ANN_model)
predANN = compute(ANN_model, test_data)
resultANN = predANN$net.result
resultANN = ifelse(resultANN > 0.5, 1, 0)
```

- library(neuralnet): Loads the neuralnet package, used for training neural networks in R.

- ANN_model = neuralnet(Class ~ ., train_data, linear.output = FALSE): Trains an artificial neural network model on the train_data to predict Class. linear.output = FALSE specifies a classification problem.

- plot(ANN_model): Plots the neural network model.

- predANN = compute(ANN_model, test_data): Computes the predictions from the neural network model on the test data.

- resultANN = predANN$net.result: Extracts the predicted probabilities from the neural network output.

- resultANN = ifelse(resultANN > 0.5, 1, 0): Converts the probabilities into binary class labels based on a threshold of 0.5.


