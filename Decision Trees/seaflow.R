library(caret)
library(ggplot2)
library(gridExtra)
library(rpart)
library(rpart.plot)
library(randomForest) 
library(e1071)

# Import data
setwd("C:/Users/drexa/git/machine-learning/")
seaflow <- read.csv("./datasets/seaflow.csv", stringsAsFactors = TRUE)
summary(seaflow)
print(table(seaflow$pop))

# Plot pe vs chl_small
dev.new() 
p_chl_small_vs_pe <- ggplot(seaflow, aes(x = seaflow$chl_small, y = seaflow$pe, color = seaflow$pop, shape = seaflow$pop)) +
                     geom_point() +
                     ggtitle("chl_small vs pe")
print(p_chl_small_vs_pe)

# Check variabe continuity
dev.new() 
p_fsc_small <- ggplot(seaflow, aes(x = seaflow$fsc_small)) +
    geom_area(stat = "bin",  fill = "lightblue")
p_fsc_perp <- ggplot(seaflow, aes(x = seaflow$fsc_perp)) +
    geom_area(stat = "bin",  fill = "lightblue")
p_fsc_big <- ggplot(seaflow, aes(x = seaflow$fsc_big)) +
    geom_area(stat = "bin",  fill = "lightblue")
p_pe <- ggplot(seaflow, aes(x = seaflow$pe)) +
    geom_area(stat = "bin",  fill = "lightblue")
p_chl_small <- ggplot(seaflow, aes(x = seaflow$chl_small)) +
    geom_area(stat = "bin",  fill = "lightblue")
p_chl_big <- ggplot(seaflow, aes(x = seaflow$chl_big)) +
    geom_area(stat = "bin",  fill = "lightblue")
grid.arrange(p_fsc_small, p_fsc_perp, p_fsc_big, p_pe, p_chl_small, p_chl_big, ncol = 3, nrow = 2)

# Remove all the data regarding file_id 208
file_id_col <- match("file_id", names(seaflow))
file_id_208 <- which(seaflow[ ,file_id_col ] == 208)
# seaflow <- seaflow[ -file_id_208, ]

# Split into training and test
set.seed(123)
trainIndex <- createDataPartition(seaflow$pop, p = .75, list = FALSE)
seaflowTrain <- seaflow[ trainIndex,]
seaflowTest  <- seaflow[-trainIndex,]

form <- formula(pop ~ fsc_small + fsc_perp + pe + chl_big + chl_small)

# Train tree
model.tree <- rpart(formula = form, method = "class", data = seaflowTrain)
print(model.tree)
# Plot tree
dev.new()
t <- rpart.plot(model.tree)
print(t)
# Feed unseen data
tree_pred = predict(model.tree, seaflowTest, type = "class")
# Confusion matrix
cm_tree <- table(pred = tree_pred, true = seaflowTest$pop)
print(cm_tree)
accuracy_tree <- sum(diag(cm_tree))/sum(cm_tree)
print(accuracy_tree)

# Train random forest
model.rd <- randomForest(form, data = seaflowTrain)
gini <- importance(model.rd)
print(gini)
# Feed unseen data
rd_pred = predict(model.rd, seaflowTest, type = "class")
# Confusion matrix
cm_rd <- table(pred = rd_pred, true = seaflowTest$pop)
print(cm_rd)
accuracy_rd <- sum(diag(cm_rd))/sum(cm_rd)
print(accuracy_rd)

# Train svm
model.svm <- svm(form, data = seaflowTrain)
# Feed unseen data
svm_pred = predict(model.svm, seaflowTest, type = "class")
# Confusion matrix
cm_svm <- table(pred = svm_pred, true = seaflowTest$pop)
print(cm_svm)
accuracy_svm <- sum(diag(cm_svm))/sum(cm_svm)
print(accuracy_svm)
