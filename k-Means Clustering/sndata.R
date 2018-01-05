library("stats")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Clustering using k-Means \n")
cat("---------------------------------------------------------- \n")

setwd("C:/Users/drexa/git/machine-learning/k-Means Clustering")

# Import data
teens <- read.csv("../datasets/sndata.csv")
cat("*** US social network profiles dataset imported \n\n")

# Missing records
cat("*** Imputating missing data \n")
teens$female <- ifelse(teens$gender == "F" & !is.na(teens$gender), 1, 0)
teens$no_gender <- ifelse(is.na(teens$gender), 1, 0)
# summary(teens$age)
# Everything not in range will be threated as missing data
teens$age <- ifelse(teens$age >= 13 & teens$age < 20, teens$age, NA)
mean(teens$age, na.rm=TRUE)
aggregate(data=teens, age~gradyear, mean, na.rm=TRUE)
ave_age <- ave(teens$age, teens$gradyear, FUN=function(x) mean(x, na.rm=TRUE))
teens$age <- ifelse(is.na(teens$age), ave_age, teens$age)
# summary(teens$age)

cat("*** Applying Z-Score standarization \n\n")
interests <- teens[5:40]
interests_z <- as.data.frame(lapply(interests, scale))

cat("*** Dividing data into 5 clusters \n\n")
set.seed(2345)
teen_clusters <- kmeans(interests_z, 5)
teen_clusters$centers

cat("*** Predicted friends \n")
print(aggregate(data=teens, friends ~ teen_clusters$cluster, mean))











