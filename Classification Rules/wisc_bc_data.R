library("RWeka")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using Decision Rules \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/machine-learning/Classification Rules")

# Import data
wbcd <- read.csv("../datasets/wisc_bc_data.csv", stringsAsFactors = FALSE)
cat("*** Wisconsin General Hospital breast cancer dataset imported \n\n")

wbcd$diagnosis <- factor(wbcd$diagnosis)

# 1 Rule classifier
cat("*** 1 Rule classifier \n\n")
wbcd_1R <- OneR(diagnosis~., data=wbcd)
print(wbcd_1R)
print(summary(wbcd_1R))

# RIPPER classifier
cat("\n*** RIPPER classifier \n\n")
wbcd_JRip <- JRip(diagnosis~., data=wbcd)
print(wbcd_JRip)
print(summary(wbcd_JRip))
