library("ggplot2")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Numeric forecasting using Linear Regression \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/R/MachineLearning/Linear Regression")
# setwd("C:/Users/dr186049/git/MachineLearning/Linear Regression")

# Import data
launch <- read.csv("../datasets/challenger.csv")
cat("*** Challenger Space Shuttle launch failure dataset imported \n\n")

# O-ring distresses detected, as compared to the temperature at launch

