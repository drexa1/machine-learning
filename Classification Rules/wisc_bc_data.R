library("RWeka")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using Decision Rules \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/R/MachineLearning/Classification Rules")
# setwd("C:/Users/dr186049/git/MachineLearning/Classification Rules")

# Import data
wbcd <- read.csv("../datasets/wisc_bc_data.csv", stringsAsFactors = FALSE)
cat("*** Wisconsin General Hospital breast cancer dataset imported \n\n")

wbcd$diagnosis <- factor(wbcd$diagnosis)

# 1 Rule classifier
wbcd_1R <- OneR(diagnosis~., data=wbcd)
print(wbcd_1R)
summary(wbcd_1R)

# RIPPER classifier
wbcd_JRip <- JRip(diagnosis~., data=wbcd)
print(wbcd_JRip)
summary(wbcd_JRip)