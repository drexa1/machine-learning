library("ggplot2")
library("rpart")
library("rpart.plot")
library("RWeka")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Numeric forecasting using Regression Trees \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/R/MachineLearning/Regression Trees")
# setwd("C:/Users/dr186049/git/MachineLearning/Regression Trees")

prompt_num <- function(str) {
    input <- readline(str)
    if(input=="") {
        return (-1)
    }
    input <- ifelse(grepl("\\d", input), as.integer(input), -1)
    if(input < 1) {
        prompt_num(str)
    }
    return (as.integer(input))
}

# Import data
wines <- read.csv("../datasets/whitewines.csv")
cat("*** UCI Vinho Verde samples dataset imported \n\n")

cat("*** Qualities overview: \n")
print(summary(wines$quality))
dev.off
ggplot(wines, aes(quality)) +
    geom_histogram(color="black", fill="light blue")

cat("\n")
n_rows <- prompt_num("*** Select number of testing rows or [ENTER] for default (20%): ")
if(n_rows == -1) {
    n_train_rows <- round(nrow(wines)*80/100)
} else {
    n_train_rows <- nrow(wines)-n_rows
}

# Split dataframe for training(80%) and model testing(20%)
cat("\n*** Splitting dataset for training and model testing\n\n")
wines_train <- wines[1:n_train_rows, ]
wines_test <- wines[(n_train_rows+1):(nrow(wines)), ]

cat("*** Building Regression Tree: \n")
model.rpart <- rpart(quality~., data=wines_train)
print(model.rpart)
rpart.plot(model.rpart, digits=4, fallen.leaves=TRUE, type=3, extra=101)

pred.rpart <- predict(model.rpart, wines_test)
print(summary(pred.rpart))

cat("\n*** Performance measurement by correlation: ")
pred_cor <- cor(pred.rpart, wines_test$quality)
cat(pred_cor)
cat(ifelse((pred_cor > 0.5) || (pred_cor < -0.5), " (strong)", " (weak)"))

MAE <- function(actual, predicted) { 
    mean(abs(actual - predicted)) 
}
cat("\n*** Performance measurement by Mean Absolute Error: ")
pred_mae <- MAE(pred.rpart, wines_test$quality)
cat(pred_mae)
cat(ifelse((pred_mae > 0.5) || (pred_mae < -0.5), " (strong)", " (weak)"))

cat("\n\n*** Building Model Tree: \n")
model.m5p <- M5P(quality~., data=wines_train)
print(summary(model.m5p))

cat("\n*** Performance measurement by correlation: ")
pred.m5p <- predict(model.m5p, wines_test)
pred_cor2 <- cor(pred.m5p, wines_test$quality)
cat(pred_cor2)
cat(ifelse((pred_cor2 > 0.5) || (pred_cor2 < -0.5), " (strong)", " (weak)"))

cat("\n*** Performance measurement by Mean Absolute Error: ")
pred_mae2 <- MAE(pred.m5p, wines_test$quality)
cat(pred_mae2)
cat(ifelse((pred_mae2 > 0.5) || (pred_mae2 < -0.5), " (strong)", " (weak)"))
