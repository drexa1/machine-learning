library("ggplot2")
library("gridExtra")
library("C50")
library("gmodels")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using Decision Trees \n")
cat("---------------------------------------------------------- \n")

# setwd("C:/Users/drexa/git/R/MachineLearning/Decision Trees")
setwd("C:/Users/dr186049/git/MachineLearning/Decision Trees")

# Import data (majority of nominal features)
credits <- read.csv("../datasets/credit.csv", stringsAsFactors = TRUE)
cat("*** Hamburg Credit agency loans dataset imported \n\n")

# Recode $default as a factor
credits$default <- factor(credits$default, c("1", "2"), c("no", "yes"))

cat("*** Checking balances: ")
print(table(credits$checking_balance))
cat("*** Savings balances: ")
print(table(credits$savings_balance))

# ggplot2 does not assume a single vector
p1 <- ggplot(credits, aes(x = 1, y = months_loan_duration)) +
             geom_boxplot(fill = "lavender") +
             labs(y = "Months loan duration") +
             theme_dark()
p2 <- ggplot(credits, aes(x = 1, y = amount)) +
             geom_boxplot(fill = "lavender") +
             labs(y = "Amount") +
             theme_dark()
grid.arrange(p1, p2, ncol = 2, nrow = 1)

set.seed(123)
train_sample <- sample(1000, 900)

credit_train <- credits[train_sample, ]
credit_test <- credits[-train_sample, ]

cat("\n*** Representativity of the samples:\n")
print(prop.table(table(credit_train$default)))
print(prop.table(table(credit_test$default)))

# Remove the result column from the prediction
resultcol_idx <- match("default", names(credit_train))
# Number of separate decision trees to use in the boosted team
# Standard value 10 is stimated to reduce up to 25% error rates
credit_model <- C5.0(credit_train[-resultcol_idx], credit_train$default)
# print(summary(credit_model))

# Results
credit_pred <- predict(credit_model, credit_test)
cat("\n*** Tree results:")
CrossTable(credit_test$default, credit_pred, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, 
           dnn=c('actual default', 'predicted default'))

# Number of separate decision trees to use in the boosted team
# Standard value 10 is stimated to reduce up to 25% error rates
credit_boost10 <- C5.0(credit_train[-resultcol_idx], credit_train$default, trials = 10)
# print(summary(credit_boost10))

credit_boost10_pred <- predict(credit_boost10, credit_test)
cat("\n*** Boosted tree results:")
CrossTable(credit_test$default, credit_boost10_pred, 
           prop.chisq=FALSE, prop.c=FALSE, prop.r = FALSE, 
           dnn=c('actual default', 'predicted default'))

# Costs matrix
matrix_dimensions <- list(c("no", "yes"), c("no", "yes"))
names(matrix_dimensions) <- c("predicted", "actual")
error_costs <- matrix(c(0, 1, 3, 0), nrow = 2, dimnames = matrix_dimensions)

credit_costs <- C5.0(credit_train[-resultcol_idx], credit_train$default, costs = error_costs)
# print(summary(credit_costs))

credit_costs_pred <- predict(credit_costs, credit_test)
cat("\n*** Tree with costs matrix results:")
CrossTable(credit_test$default, credit_costs_pred, 
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, 
           dnn=c('actual default', 'predicted default'))

