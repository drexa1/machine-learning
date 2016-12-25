library("kernlab")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Modeling using Support Vector Machines \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/R/MachineLearning/Support Vector Machines")
# setwd("C:/Users/dr186049/git/MachineLearning/Support Vector Machines")

# Import data
letters <- read.csv("../datasets/letterdata.csv")
cat("*** UCI English alphabet dataset imported \n\n")

cat("\n*** Splitting dataset for training and model testing\n\n")
n_train_rows <- round(nrow(letters)*80/100)
letters_train <- letters[1:n_train_rows, ]
letters_test <- letters[(n_train_rows+1):(nrow(letters)), ]

classifier <- ksvm(letter~., data=letters_train, kernel="vanilladot")


