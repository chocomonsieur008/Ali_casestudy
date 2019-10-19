# Set options and environment variables -----------------------------------
# Clean the memory 
remove(list = ls())

# Release memory from nuisance objects
gc()

# Switch off the scientific notation
options(scipen = 999)

# Switch off strings to factor conversion
options(stringsAsFactors = FALSE)

# Set max print
options(max.print = 1000)

# Set timezone
Sys.setenv(TZ = 'UTC')

# Set Java JDK location
Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk-13")
Sys.getenv("JAVA_HOME")


# Load required pacakges --------------------------------------------------
# Remmeber the dependencies of the package caret are needed
# install.packages("caret", dependencies = TRUE)

library(dplyr)
library(data.table)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggcorrplot)
library(rpart)
library(rpart.plot)
library(caret)
library(rJava)
library(FSelector)
library(randomForest)
library(glmnet)
library(fitdistrplus)
library(pROC)
library(ROCR)


# Set working directory ---------------------------------------------------

root.dir <- "C:/Users/Ali/Documents/R/Ali_casestudy"


# Source functions --------------------------------------------------------

source(file = file.path(root.dir, "utility_functions.R"))


# Read training and testing set -------------------------------------------

training.set <- data.table::fread(file.path(root.dir, "data", "train.csv"))
training.set <- training.set[ , -c(1,2)]

# Rename the column name of the response variable
colnames(training.set)[31] <- "class"

training.set$class <- as.factor(training.set$class)

testing.set.org <- data.table::fread(file.path(root.dir, "data", "test.csv"))
testing.set.index <- testing.set.org[ , 2]
testing.set <- testing.set.org[ , -c(1,2)]


# Check descriptive statistics --------------------------------------------

str(training.set)
summary(training.set)

str(testing.set)
summary(testing.set)

# Principle Component Analysis --------------------------------------------
# Some tryouts althought the intuitive relation between variables are unknown

training.set.cor <- stats::cor(x = training.set[, -31], method = "pearson")

ggcorrplot::ggcorrplot(training.set.cor, type = "lower")

# PCA
training.set.pca <- stats::prcomp(x = training.set[, -31], center = TRUE, scale. = TRUE)

summary(training.set.pca)

# Plot PCAs
variance.ex <- tibble(sdev = training.set.pca$sdev)
variance.ex$pca.comp <- 1:nrow(variance.ex)

variance.ex <- variance.ex %>% 
  mutate(pc.variance = sdev^2,
         prop.variance.ex = pc.variance/sum(pc.variance))

ggplot2::ggplot(variance.ex, aes(pca.comp, prop.variance.ex, group = 1)) +
  geom_line() + 
  geom_point()

# Result: inconclusive

# Logistic regression -----------------------------------------------------

# Logit model with backward elimination
# Full model
logit.model.full <- glm(formula = class ~ ., data = training.set, family = "binomial")

summary(logit.model.full)
get_mc_fadden_R2(logit.model = logit.model.full)

# Backward Elimination
logit.formula <- construct_reg_formula(data = training.set, cols = c(3:4, 8:10, 13:14, 20:22, 27))

logit.model.backward <- glm(formula = logit.formula, data = training.set, family = "binomial")

summary(logit.model.backward)
get_mc_fadden_R2(logit.model = logit.model.backward)


# Logit model with forward selection
# Null model
logit.model.null <- glm(formula = class ~ 1, data = training.set, family = "binomial")

summary(logit.model.null)
get_mc_fadden_R2(logit.model = logit.model.null)

# Forward Selection
logit.formula <- construct_reg_formula(data = training.set, cols = c(3:4, 8, 10, 13:14, 16, 22))

logit.model.forward <- glm(formula = logit.formula, data = training.set, family = "binomial")

summary(logit.model.forward)
get_mc_fadden_R2(logit.model = logit.model.forward)


# Prediction preformance
# 1. AUC
roc.full <- roc(response = training.set$class, 
                predictor = logit.model.full$fitted.values, 
                plot = TRUE, 
                print.auc = TRUE, 
                xlab = "True Negative Rate", 
                ylab = "True Positive Rate", 
                col = "#377eb8", lwd = 4)

roc.full$auc

roc.backward <- roc(response = training.set$class, 
                    predictor = logit.model.backward$fitted.values, 
                    plot = TRUE,
                    print.auc = TRUE, 
                    xlab = "True Negative Rate", 
                    ylab = "True Positive Rate", 
                    col = "#ff5733", lwd = 4)

roc.backward$auc

roc.null <- roc(response = training.set$class, 
                predictor = logit.model.null$fitted.values, 
                plot = TRUE,
                print.auc = TRUE, 
                xlab = "True Negative Rate", 
                ylab = "True Positive Rate", 
                col = "#1ea43e", lwd = 4)

roc.null$auc

roc.forward <- roc(response = training.set$class, 
                   predictor = logit.model.forward$fitted.values, 
                   plot = TRUE,
                   print.auc = TRUE, 
                   xlab = "True Negative Rate", 
                   ylab = "True Positive Rate", 
                   col = "#ca4bc8", lwd = 4)

roc.forward$auc

# 2. Alternatively, Concordance and Discordance
concordance.discordance.full <- get_concordance_discordance(logit.model.full)
concordance.discordance.backward <- get_concordance_discordance(logit.model.backward)
concordance.discordance.null <- get_concordance_discordance(logit.model.null)
concordance.discordance.forward <- get_concordance_discordance(logit.model.forward)


# Prediction with the selected model
testing.set$predicted.log.odds.class <- predict(logit.model.forward, testing.set, type = "link")
testing.set$predicted.prob.class <- exp(testing.set$predicted.log.odds.class) / (1 + exp(testing.set$predicted.log.odds.class))
testing.set$predicted.class <- ifelse(testing.set$predicted.prob.class > 0.5, 1, 0)
testing.set$index <- testing.set.index


# Store the output
data.table::fwrite(x = testing.set,
                   file = file.path(root.dir, "results", "logit_result.csv"), row.names = FALSE, quote = FALSE)


# Decision tree -----------------------------------------------------------

testing.set <- testing.set.org[ , -c(1,2)]

# Decision tree with backward elimination
# Full model
trained.decision.tree.full <- rpart::rpart(formula = class ~ ., data = training.set, method = "class")

# Take a look at the constructed tree
rpart.plot::prp(x = trained.decision.tree.full)

# Get the confusion matrix
training.prediction.full <- stats::predict(trained.decision.tree.full, training.set, type = "class")
caret::confusionMatrix(data = training.prediction.full, reference = training.set$class)


# Backward elimination
decision.formula <- construct_reg_formula(data = training.set, cols = c(3:4, 8:10, 13:14, 20:22, 27))

trained.decision.tree.backward <- rpart::rpart(formula = decision.formula, data = training.set, method = "class")

# Take a look at the constructed tree
rpart.plot::prp(x = trained.decision.tree.backward)

# Get the confusion matrix
training.prediction.backward <- stats::predict(trained.decision.tree.backward, training.set, type = "class")
caret::confusionMatrix(data = training.prediction.backward, reference = training.set$class)


# Decision tree with forward selection
decision.formula <- construct_reg_formula(data = training.set, cols = c(3:4, 8, 10, 13:14, 16, 22))

trained.decision.tree.forward <- rpart::rpart(formula = decision.formula, data = training.set, method = "class")

# Take a look at the constructed tree
rpart.plot::prp(x = trained.decision.tree.forward)

# Get the confusion matrix
class.prediction.forward <- stats::predict(trained.decision.tree.forward, training.set, type = "class")
caret::confusionMatrix(data = class.prediction.forward, reference = training.set$class)


# Prediction with the selected model
class.prediction.forward <- stats::predict(trained.decision.tree.forward, testing.set, type = "class")

testing.set$predicted.class <- class.prediction.forward
testing.set$index <- testing.set.index

# Store the output
data.table::fwrite(x = testing.set,
                   file = file.path(root.dir, "results", "decision_result.csv"), row.names = FALSE, quote = FALSE)


# Random forest -----------------------------------------------------------
# Some tryouts althought the the outcome of decision trees are accurate enough
testing.set <- testing.set.org[ , -c(1,2)]

# Random forest with backward elimination
# Full model
rf.full <- randomForest::randomForest(formula = class ~ ., data = training.set, ntree = 36, importance = TRUE)

# Get the confusion matrix
class.prediction.forest <- stats::predict(rf.full, training.set, type = "class")
caret::confusionMatrix(data = class.prediction.forest, reference = training.set$class)


# Backward elimination
forest.formula.backward <- construct_reg_formula(data = training.set, cols = c(3:4, 8:10, 13:14, 20:22, 27))

rf.backward <- randomForest::randomForest(formula = forest.formula.backward, data = training.set, ntree = 36, importance = TRUE)

# Get the confusion matrix
class.prediction.forest <- stats::predict(rf.backward, training.set, type = "class")
caret::confusionMatrix(data = class.prediction.forest, reference = training.set$class)


# Forward selection
forest.formula.forward <- construct_reg_formula(data = training.set, cols = c(3:4, 8, 10, 13:14, 16, 22))

rf.forward <- randomForest::randomForest(formula = forest.formula.forward, data = training.set, ntree = 36, importance = TRUE)

# Get the confusion matrix
class.prediction.forest <- stats::predict(rf.forward, training.set, type = "class")
caret::confusionMatrix(data = class.prediction.forest, reference = training.set$class)


# Prediction with the selected model
class.prediction.rf <- stats::predict(rf.forward, testing.set, type = "response")

testing.set$predicted.class <- class.prediction.rf
testing.set$index <- testing.set.index

# Store the output
data.table::fwrite(x = testing.set,
                   file = file.path(root.dir, "results", "forest_result.csv"), row.names = FALSE, quote = FALSE)



