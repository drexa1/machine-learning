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

cat("*** Splitting dataset for training and model testing\n\n")
n_train_rows <- round(nrow(letters)*80/100)
letters_train <- letters[1:n_train_rows, ]
letters_test <- letters[(n_train_rows+1):(nrow(letters)), ]

cat("*** Building SVM with linear kernel \n")
classifier <- ksvm(letter~., data=letters_train, kernel="vanilladot")
print(classifier)

predicted_letters <- predict(classifier, letters_test)
agreement <- predicted_letters == letters_test$letter
cat("\n*** ")
print(prop.table(table(agreement)))

cat("\n*** Building SVM with gaussian kernel \n")
rbf_classifier <- ksvm(letter~., data=letters_train, kernel="rbfdot")
print(rbf_classifier)

rbf_predicted_letters <- predict(rbf_classifier, letters_test)
rbf_agreement <- rbf_predicted_letters == letters_test$letter
cat("\n*** ")
print(prop.table(table(rbf_agreement)))
