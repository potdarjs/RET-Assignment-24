
# ----------------------------------- Assignment 24 part 2 ----------------------- 

# Load SparkR library into your R session
library(SparkR)

# Initialize SparkSession
sparkR.session(appName = "SparkR-ML-logit-example")

# Binomial logistic regression

# $example on:binomial$
# Load training data
df <- read.df("libsvm_data.txt", source = "libsvm")
training <- df
test <- df

# Fit an binomial logistic regression model with spark.logit
model <- spark.logit(training, label ~ features, maxIter = 10, regParam = 0.3, elasticNetParam = 0.8)

# Model summary
summary(model)

# Prediction
predictions <- predict(model, test)
head(predictions)
# $example off:binomial$

# Multinomial logistic regression

# $example on:multinomial$
# Load training data
df <- read.df("multiclass_classification_data.txt", source = "libsvm")
training <- df
test <- df

# Fit a multinomial logistic regression model with spark.logit
model <- spark.logit(training, label ~ features, maxIter = 10, regParam = 0.3, elasticNetParam = 0.8)

# Model summary
summary(model)

# Prediction
predictions <- predict(model, test)
head(predictions)
# $example off:multinomial$

sparkR.session.stop()
