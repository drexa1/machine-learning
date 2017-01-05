library("arules")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Pattern discovery using Association Rules \n")
cat("---------------------------------------------------------- \n")

# setwd("C:/Users/drexa/git/R/MachineLearning/Association Rules")
setwd("C:/Users/dr186049/git/MachineLearning/Association Rules")

# Import data
groceries_csv <- read.csv("../datasets/groceries.csv")
cat("*** ARules package groceries basket dataset imported \n\n")

cat("*** Creating sparse matrix of transactions \n")
groceries <- read.transactions("../datasets/groceries.csv", sep=",")
print(summary(groceries))

cat("\n*** Plotting items with support = 10% ")
itemFrequencyPlot(groceries, support=0.1)
cat("\n*** Plotting top 20 items")
itemFrequencyPlot(groceries, topN=20)
cat("\n*** Plotting sparse matrix for random sample ")
# columns heavily populated indicate popular items
print(image(sample(groceries, 200)))

cat("\n\n*** Generating A Priori ruleset ")
# support threshold=0.006: items purchased 2 a day > 60 times per month > 60% out of 9835 trans.
# confidence=0.25: the rule has to be correct 25% of the time
# minlen=2: rules of at least 2 items
groceries_rules <- apriori(groceries, parameter=list(support=0.006, confidence=0.25, minlen=2))
print(summary(groceries_rules))

cat("\n*** Listing top 5 rules \n\n")
inspect(sort(groceries_rules, by="lift")[1:5])
cat("\n*** Listing rules concerning 'berries' \n\n")
berryrules <- subset(groceries_rules, items %in% "berries")
inspect(berryrules)

# saving rules to dataset
cat("\n*** Saving rules to dataset \n")
groceries_rules_df <- as(groceries_rules, "data.frame")
# saving rules to CSV
cat("*** Saving rules to CSV \n")
write(groceries_rules, file = "groceries_rules.csv", sep = ",", quote = TRUE, row.names = TRUE)
