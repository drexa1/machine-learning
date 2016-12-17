library("class")
library("gmodels")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" Classification using k-Nearest Neighbors \n")
cat("---------------------------------------------------------- \n")

# Import data
setwd("C:/Users/drexa/git/R/MachineLearning/k-Nearest Neighbors")
# setwd("C:/Users/dr186049/git/MachineLearning/k-Nearest Neighbors")
wbcd <- read.csv("../datasets/wiscbcdata.csv", stringsAsFactors = FALSE)
cat("*** Wisconsin General Hospital breast cancer dataset imported \n\n")

pause <- function() {
    (prompt="Press [enter] to continue \n")
}
prompt_ys <- function(str) {
    input <- readline(str)
    input <- ifelse(grepl("y|n", input), as.character(input), -1)
    if(!is.character(input)) {
        prompt_norm(str)
    }
    return (as.character(input))
}
prompt_norm <- function(str) {
    input <- readline(str)
    input <- ifelse(grepl("n|z", input), as.character(input), -1)
    if(!is.character(input)) {
        prompt_norm(str)
    }
    return (as.character(input))
}
prompt_feature <- function(str) {
    input <- readline(str)
    if(!is.element(input, names(wbcd_norm))) {
        return (-1)
    }
    return (as.character(input))
}
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

# Exclude ID to avoid spurius predictions
id_idx <- match("id", names(wbcd))
wbcd <- wbcd[-id_idx]
cat("*** Removed patient PK to avoid overfitting \n")
pause()

# Recode diagnosis as a factor
# table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis, c("B", "M"), c("Benign", "Malignant"))
cat("*** Recoded $diagnosis as factor \n")
pause()

# Show percentages
cat("*** Percentages for the diagnosis factor \n")
print(round(prop.table(table(wbcd$diagnosis))*100, digits=2))
pause()

# Some numeric features hold way larger measures than others
cat("\n*** Normalization is required \n")
print(summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")]))
pause()

# Create normalization function
normalize <- function(x) {
    return ((x-min(x)) / (max(x)-min(x)))
}
# Normalize numeric features
confirm <- prompt_norm("*** Select 0-1 standard normalization or z-scale standardization (n/z): ")
if(confirm == "n") {
    # Disperse numeric features now range from 0 to 1
    wbcd_norm <- as.data.frame(lapply(wbcd[2:ncol(wbcd)], normalize))
} else if(confirm == "z") {
    # The rescaled numeric features have now a mean zero
    wbcd_norm <- as.data.frame(scale(wbcd[-1]))
}

# Show rescaled features
feature_name <- prompt_feature("*** Select feature to inspect after rescale or [ENTER] to skip: ")
if(is.element(feature_name, names(wbcd_norm))) {
    print(summary(wbcd_norm[ , feature_name])) 
}

# cat("\n *** Shuffling rows")
# wbcd_rand <- wbcd_norm[sample(1:nrow(wbcd_norm)), ]
# No randomizing
cat("*** No randomizing")
wbcd_rand <- wbcd_norm

# Split dataframe for training(80%) and model testing(20%)
cat("\n *** Splitting dataset for training and model testing")
n_rows <- prompt_num("*** Select number of testing rows or [ENTER] for default (20%): ")
if(n_rows == -1) {
    n_train_rows <- round(nrow(wbcd_rand)*80/100)
} else {
    n_train_rows <- nrow(wbcd_rand)-n_rows
}
message("*** Train rows: ", n_train_rows)
wbcd_train <- wbcd_rand[1:n_train_rows, ]
message("*** Test rows: ", (nrow(wbcd_rand)-n_train_rows))
wbcd_test <- wbcd_rand[(n_train_rows+1):(nrow(wbcd_rand)), ]
pause()

# Extracting diagnosis factor for perfomance comparison
cat("*** Extracting $diagnosis factor for perfomance comparison \n")
wbcd_train_labels <- wbcd[1:n_train_rows, 1]
wbcd_test_labels <- wbcd[(n_train_rows+1):(nrow(wbcd_rand)), 1]
pause()

# Find an optimus K value
get_k_val <- function() {
    k_val <- round(sqrt(n_train_rows))
    k_val <- ifelse((k_val%%2)!=0, k_val, k_val+1)
    return (k_val)
}

k_val <- prompt_num("*** Select value for K or [ENTER] for default: ")
if(k_val == -1) {
    k_val <- get_k_val()
}
message("*** Performing knn classification with k=", k_val)
wbcd_pred <- knn(train=wbcd_train, test=wbcd_test, cl=wbcd_train_labels, k=k_val)

# Results
CrossTable(x=wbcd_test_labels, y=wbcd_pred, prop.chisq=FALSE)
