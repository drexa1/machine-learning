library("tm")
library("wordcloud")
library("SnowballC")
library("e1071")

library("caret")
library("irr")
library("ROCR")

options(warn = -1)
set.seed(123)

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using Naive Bayes (Performance evaluation) \n")
cat("---------------------------------------------------------- \n")

# Import data
# setwd("C:/Users/drexa/git/R/MachineLearning/Naive Bayes")
setwd("C:/Users/dr186049/git/MachineLearning/Naive Bayes")
sms_raw <- read.csv("../datasets/sms_spam.csv", stringsAsFactors = FALSE)
cat("*** SMS Spam Collection dataset imported \n\n")

# Convert type to factor
sms_raw$type <- factor(sms_raw$type)
cat("*** Recoded $type as factor \n")

# Create volative corpus
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
cat("*** Generated volatile corpus \n")

# Apply transformations
cat("\n*** Generating terms matrix from document... \n")
sms_dtm <- DocumentTermMatrix(sms_corpus, control = list(tolower = TRUE, 
                                                         removeNumbers = TRUE, 
                                                         stopwords = TRUE, 
                                                         removePunctuation = TRUE, 
                                                         stemming = TRUE))

# Holdout sampling
cat("*** Creating balanced partitions for training and testing \n")
in_train <- createDataPartition(sms_raw$text, times = 1, p = 0.75, list = FALSE)
train <- sms_dtm[in_train, ]
test <- sms_dtm[-in_train,]

# Extracting type factor for perfomance comparison
cat("*** Extracting $type for perfomance comparison \n\n")
train_labels <- sms_raw[index, ]$type
test_labels <- sms_raw[-index, ]$type

# Confirm representativity of the testing subset
cat("*** Representativity of the testing subset: \n\n")
print(prop.table(table(train_labels)))
cat("\n")
print(prop.table(table(test_labels)))

# Reduce the sparse matrix
cat("\n*** Reducing training and test sparse DTMs to frequent terms \n")
freq_words <- findFreqTerms(train, 5)
freq_train <- train[ ,freq_words]
freq_test <- test[ ,freq_words]

# Turn columns into categorical
cat("*** Convert DTMs to categorical \n")
convert_counts <- function(x) { 
    x <- ifelse(x > 0, "YES", "NO") 
}
sms_train <- apply(freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(freq_test, MARGIN = 2, convert_counts)

# Enable Laplace estimator
classifier <- naiveBayes(sms_train, train_labels, laplace = 1)
cat("*** Classifier build, predicting... \n\n")
test_pred <- predict(classifier, sms_test)

# Caret results
results <- read.csv("../datasets/sms_results.csv", stringsAsFactors = FALSE)
caret_conf_matrix <- confusionMatrix(results$predict_type, 
                                     results$actual_type, 
                                     positive = "spam")
# print(caret_conf_matrix)
cat("*** Severe class imbalance, real possibility of a correct prediction by chance alone: \n")
print(caret_conf_matrix$overall[2])
cat("*** Alternative Kappa: ")
irr.kappa <- kappa2(results[1:2])
print(unlist(irr.kappa$value))

# ROC visualizing
roc_pred <- prediction(results$prob_spam, results$actual_type)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC curve for current classifier", col = "lavender", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
cat("*** AUC: ")
roc_perf.uac <- performance(roc_pred, measure = "auc")
print(unlist(roc_perf.uac@y.values))
