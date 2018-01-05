library("RWeka")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using Decision Rules \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/machine-learning/Classification Rules")

# Import data (majority of nominal features)
mushrooms <- read.csv("../datasets/mushrooms.csv", stringsAsFactors = TRUE)
cat("*** Carnegie Mellon University Mushrooms dataset imported \n\n")

# Since the factor for this feature is always the same level, it does not provide any info
# (Asign to NULL removes the feature from the data frame)
mushrooms$veil_type <- NULL

# We don't need a test set since we are not trying to cover unforeseen cases,
# but classifying the whole exisiting set

# 1 Rule classifier
cat("*** 1 Rule classifier \n\n")
mushroom_1R <- OneR(type~., data=mushrooms)
print(mushroom_1R)
print(summary(mushroom_1R))

# RIPPER classifier
cat("\n*** RIPPER classifier \n\n")
mushroom_JRip <- JRip(type~., data=mushrooms)
print(mushroom_JRip)
print(summary(mushroom_JRip))
