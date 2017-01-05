library("ggplot2")
library("psych")
library("corrplot")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Numeric forecasting using Linear Regression \n")
cat("---------------------------------------------------------- \n")

# setwd("C:/Users/drexa/git/R/MachineLearning/Linear Regression")
setwd("C:/Users/dr186049/git/MachineLearning/Linear Regression")

# Import data
insurance <- read.csv("../datasets/insurance.csv", stringsAsFactors = TRUE)
cat("*** Hypothetical US patients medical expenses dataset imported \n\n")

cat("*** Expenses overview: \n")
print(summary(insurance$charges))

# Mean value is substantially greater than the median
insurance$high_rev <- as.factor(insurance[,7] < mean(insurance$charges))
ggplot(insurance, aes(charges, fill=high_rev)) +
       geom_histogram(color="black") +
       xlab("Expenses") +
       ggtitle("Distribution of expenses")

cat("\n*** Correlation among numeric features: \n")
# diagonal is always 1 (perfect correlation with the same feature itself)
cor_matrix <- cor(insurance[c("age", "bmi", "children", "charges")])
print(cor_matrix)
pairs.panels(insurance[c("age", "bmi", "children", "charges")], pch=19, lm=TRUE)
corrplot(cor_matrix, method="color")

# Train linear model
cat("\n*** Training expenses linear model: \n")
ins_model <- lm(charges ~ age+children+bmi+sex+smoker+region, data=insurance)
# Intercept is the value of the predicted feature where independent features = 0
# (this will very unlikely happen)
# Interpret the coefficients of the dummy variables in relation to the reference category
print(ins_model)

cat("*** Summary for expenses linear model: \n")
# For model performance observe R-squared value
# (this model explain r% of the variation in the dependent variable)
print(summary(ins_model))

cat("*** Appending a 2nd $age as non linear impact relationship \n")
# will trigger generation of non-linear beta
insurance$age2 <- insurance$age^2
cat("*** Transforming bmi to binary indicator \n")
# indicates obesity
insurance$bmi30 <- ifelse(insurance$bmi >= 30, 1, 0)

cat("\n*** Training expenses model with interaction effects: \n")
impr_ins_model <- lm(charges ~ age+age2+children+bmi+bmi30+bmi*smoker+sex+region, data=insurance)
print(impr_ins_model)
cat("*** Summary for improved expenses model: \n")
print(summary(impr_ins_model))
