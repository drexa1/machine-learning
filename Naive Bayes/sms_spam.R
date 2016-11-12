library("tm")
library("wordcloud")
library("e1071")
library("gmodels")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using Naive Bayes \n")
cat("---------------------------------------------------------- \n\n")

# Import data
setwd("C:/Users/drexa/git/R/MachineLearning/Naive Bayes")
sms_raw <- read.csv("../datasets/sms_spam.csv", stringsAsFactors = FALSE)
cat("*** SMS Spam Collection dataset imported \n")

prompt_num <- function(str) {
    input <- readline(str)
    if(input=="") {
        return (-1)
    }
    input <- ifelse(grepl("\\d", input), as.integer(input), -1)
    if(input < 1) {
        prompt_num(str)
    }
    return (as.integer(input))
}
prompt_ys <- function(str) {
    input <- readline(str)
    input <- ifelse(grepl("y|n", input), as.character(input), -1)
    if(!is.character(input)) {
        prompt_norm(str)
    }
    return (as.character(input))
}

# Convert type to factor
sms_raw$type <- factor(sms_raw$type)
cat("*** Recoded $type as factor \n")

# Create volative corpus
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
cat("*** Generated volatile corpus \n")

# Apply transformations manually (order matters)
# sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
# sms_corpus_clean <- tm_map(sms_corpus, removePunctuation)
# sms_corpus_clean <- tm_map(sms_corpus, removeNumbers)
# sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
# Irrelevant words
# sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())
# Stemming
# sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
# cat("*** Corpus cleaned \n")

example <- sample(1:length(sms_corpus), 1)
cat(as.character(sms_corpus[[example]]))
cat("\n")
cat(as.character(sms_corpus_clean[[example]]))

# Tokenize into term matrix
# sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
# Alternative order of transformations
if(FALSE)
sms_dtm <- DocumentTermMatrix(sms_corpus, 
                control = list(tolower = TRUE, 
                               removeNumbers = TRUE, 
                               stopwords = TRUE, 
                               removePunctuation = TRUE, 
                               stemming = TRUE))
cat("\n\n *** Term matrix created \n")

# Split dataframe for training(75%) and model testing(25%)
cat("*** Splitting dataset for training and model testing")
n_rows <- prompt_num("*** Select number of testing rows or [ENTER] for default (25%): ")
if(n_rows == -1) {
    n_train_rows <- round(nrow(sms_dtm)*75/100)
} else {
    n_train_rows <- nrow(sms_dtm)-n_rows
}
message("*** Train rows: ", n_train_rows)
sms_dtm_train <- sms_dtm[1:n_train_rows, ]
message("*** Test rows: ", (nrow(sms_dtm)-n_train_rows))
sms_dtm_test <- sms_dtm[(n_train_rows+1):(nrow(sms_dtm)), ]

# Extracting type factor for perfomance comparison
cat("*** Extracting $type factor for perfomance comparison \n")
sms_train_labels <- sms_raw[1:n_train_rows, ]$type 
sms_test_labels <- sms_raw[(n_train_rows+1):(nrow(sms_dtm)), ]$type

# Confirm representativity of the testing subset
print(prop.table(table(sms_train_labels)))
print(prop.table(table(sms_test_labels)))

# Subset for wordclouds
spam <- subset(sms_raw, type=="spam")
ham <- subset(sms_raw, type=="ham")
cat("\n*** Subset ham/spam for wordclouds \n")

# 50 represents 1% of the whole dataset
par(mfrow=c(1,2)) 
wordcloud(spam$text, max.words=50, random.order=FALSE, scale=c(2,0.3))
wordcloud(ham$text, max.words=50, random.order=FALSE, scale=c(2,0.3))

# Reduce the sparse matrix
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]
cat("*** Training and test sparse dtms reduced \n")

# Turn columns into categorical
convert_counts <- function(x) { 
    x <- ifelse(x>0, "YES", "NO") 
}
sms_train <- apply(sms_dtm_freq_train, MARGIN=2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN=2, convert_counts)
cat("*** DTMs converted to categorical \n")

# Enable Laplace estimator
confirm <- prompt_ys("*** Enable Laplace estimator (y/n): ")
laplace_estimator <- ifelse(confirm=="y", 1, 0)
sms_classifier <- naiveBayes(sms_train, sms_train_labels, laplace=laplace_estimator)
cat("*** Classifier build, predicting... \n")
sms_test_pred <- predict(sms_classifier, sms_test)

# Results
print(CrossTable(sms_test_pred, sms_test_labels, 
           prop.chisq=FALSE, 
           prop.t=FALSE, 
           dnn = c('Predicted', 'Actual')))

