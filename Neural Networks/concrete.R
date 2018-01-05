library("neuralnet")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Modeling using Neural Networks \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/machine-learning/Neural Networks")

# Import data
concrete <- read.csv("../datasets/concrete.csv")
cat("*** UCI concrete compressive strength dataset imported \n\n")

# Min-Max normalization
normalize <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
}
cat("*** Min-Max normalization \n")
norm_concrete <- as.data.frame(lapply(concrete, normalize))

cat("\n*** Splitting dataset for training and model testing \n")
n_train_rows <- round(nrow(norm_concrete)*80/100)
concrete_train <- norm_concrete[1:n_train_rows, ]
concrete_test <- norm_concrete[(n_train_rows+1):(nrow(norm_concrete)), ]

set.seed(12345)
cat("\n*** Training model by backpropagation \n")
model <- neuralnet(strength~cement+slag+ash+water+superplastic+coarseagg+fineagg+age, 
                   data=concrete_train)
plot(model)

results <- compute(model, concrete_test[1:8])
predicted_strengths <- results$net.result
cat("\n*** Correlation between predicted and actual results: ")
pred_cor <- cor(predicted_strengths, concrete_test$strength)
cat(pred_cor)

cat("\n\n*** Training improved model with hidden nodes \n")
impr_model <- neuralnet(strength~cement+slag+ash+water+superplastic+coarseagg+fineagg+age, 
                        data=concrete_train, hidden=8)
plot(impr_model)

results2 <- compute(impr_model, concrete_test[1:8])
predicted_strengths2 <- results2$net.result
cat("\n*** Correlation between predicted and actual results: ")
pred_cor2 <- cor(predicted_strengths2, concrete_test$strength)
cat(pred_cor2)
