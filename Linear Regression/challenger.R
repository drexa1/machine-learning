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
# dev.cur, 
dev.off
ggplot(launch, aes(x=temperature, y=distress_ct)) + 
       geom_point() +
       labs(x="Temperature", y="Distress events") +
       geom_smooth(method=lm, colour="red", se=TRUE)

cat("*** Estimation of 'a' and 'b' by Ordinary Least Squares \n")
cat("*** b = cov(temperature,distress)/var(temperature): ")
b <- cov(launch$temperature, launch$distress_ct) / var(launch$temperature)
cat(b)
cat("\n*** a = mean(distress) - b * mean(temperature): ")
a <- mean(launch$distress_ct) - b * mean(launch$temperature)
cat(a)

cat("\n\n*** Linearity correlation by Pearson correlation coefficient \n")
cat("*** r = cov(temperature,distress)/sd(distress): ")
r = cov(launch$temperature, launch$distress) / (sd(launch$distress) * sd(launch$temperature))
rr = cor(launch$temperature, launch$distress)
cat(r)
cat(ifelse((r > 0.5 || r < -0.5), " (strong)", " (weak)"))

cat("\n\n*** Estimate for multiple coefficients \n")
cat("*** b^ = (xT * x)inv * xT * y \n\n")
mreg <- function(y, x) {
    x <- as.matrix(x)
    x <- cbind(Intercept=1, x)
    b <- solve(t(x) %*% x) %*% t(x) %*% y
    colnames(b) <- "Estimate"
    print(b)
}
mreg(y=launch$distress_ct, x=launch[3:5])
