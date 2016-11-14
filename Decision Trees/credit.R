library("ggplot2")
library("gridExtra")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using Decision Trees \n")
cat("---------------------------------------------------------- \n")

# Import data (majority of nominal features)
setwd("C:/Users/drexa/git/R/MachineLearning/Decision Trees")
# setwd("C:/Users/dr186049/git/MachineLearning/Decision Trees")
credits <- read.csv("../datasets/credit.csv", stringsAsFactors = TRUE)
cat("*** Hamburg Credit agency loans dataset imported \n\n")

print(table(credits$checking_balance))
print(table(credits$savings_balance))

# ggplot2 does not assume a single vector
p1 <- ggplot(credits, aes(x=1, y=months_loan_duration)) +
             geom_boxplot(fill="lavender") +
             labs(y="Months loan duration") +
             theme_light()
p2 <- ggplot(credits, aes(x=1, y=amount)) +
             geom_boxplot(fill="lavender") +
             labs(y="Amount") +
             theme_light()
grid.arrange(p1, p2, ncol=2, nrow=1)


