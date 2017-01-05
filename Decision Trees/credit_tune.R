library("C50")

library("caret")

options(warn = -1)
set.seed(123)

cat("\n")
cat("------------------------------------------------------------- \n")
cat(" Classification using Decision Trees (Performance evaluation) \n")
cat("------------------------------------------------------------- \n")

# setwd("C:/Users/drexa/git/R/MachineLearning/Decision Trees")
setwd("C:/Users/dr186049/git/MachineLearning/Decision Trees")

# Import data (majority of nominal features)
credits <- read.csv("../datasets/credit.csv", stringsAsFactors = TRUE)
cat("*** Hamburg credit agency loans dataset imported \n\n")

# Recode $default as a factor
credits$default <- factor(credits$default, c("1", "2"), c("no", "yes"))

# Caret automated data munging and parameter tunning
model <- train(default ~ ., data = credits, method = "C5.0")

# Not indicative for unseen data
pred <- predict(model, credits)
print(table(pred, credits$default))