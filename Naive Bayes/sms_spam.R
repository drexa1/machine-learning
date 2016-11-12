library("tm")
library("wordcloud")

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

# Convert type to factor
sms_raw$type <- factor(sms_raw$type)
cat("*** Recoded $type as factor \n")

# Create volative corpus
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
cat("*** Generated volatile corpus \n")

# Apply transformations manually (order matters)
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
sms_corpus_clean <- tm_map(sms_corpus, removePunctuation)
sms_corpus_clean <- tm_map(sms_corpus, removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
# Irrelevant words
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())
# Stemming
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
cat("*** Corpus cleaned \n")

example <- sample(1:length(sms_corpus), 1)
cat(as.character(sms_corpus[[example]]))
cat("\n")
cat(as.character(sms_corpus_clean[[example]]))

# Tokenize into term matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
# Alternative order of transformations
if(FALSE)
sms_dtm2 <- DocumentTermMatrix(sms_corpus, 
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
cat("\n *** Subset data set for wordclouds \n")

# 50 represents 1% of the whole dataset
par(mfrow=c(1,2)) 
wordcloud(spam$text, max.words=50, random.order=FALSE, scale=c(3,0.5))
wordcloud(ham$text, max.words=50, random.order=FALSE, scale=c(3,0.5))
