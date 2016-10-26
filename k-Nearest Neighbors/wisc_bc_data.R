library("class")
library("gmodels")

cat("\n")
cat("---------------------------------------------------------- \n")
cat(" k-Nearest Neighbors lazy classification \n")
cat("---------------------------------------------------------- \n\n")

# Import data
setwd("C:/Users/drexa/git/R/Machine Learning")
wbcd <- read.csv("datasets/wisc_bc_data.csv", stringsAsFactors = FALSE)
cat(" *** Wisconsin General Hospital breast cancer dataset imported \n")

prompt <- function(str) {
    input <- readline(str)
    input <- ifelse(grepl("^y|^n?$", input), as.character(input), -1)
    if(!is.character(input)) {
        prompt(str)
    }
    return (input)
}
prompt_norm <- function(str) {
    input <- readline(str)
    input <- ifelse(grepl("^n|^z?$", input), as.character(input), -1)
    if(!is.character(input)) {
        prompt(str)
    }
    return (input)
}
prompt_str <- function(str) {
    input <- readline(str)
    if(grepl("\\s", input)) {
        break;
    } else if(!is.element(input, names(wbcd_norm))) {
        prompt(str)
    }
    return (input)
}

# Dataset structure
confirm <- prompt(" Check dataset structure? (y/n): ")
if(confirm == "y") {
    str(wbcd)
}

# Exclude ID to avoid spurius predictions
id_idx <- match("id", names(wbcd))
wbcd <- wbcd[-id_idx]
cat("\n *** Removed patient PK to avoid overfitting")

# Recode diagnosis as a factor
# table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis, c("B", "M"), c("Benign", "Malignant"))
cat("\n *** Recoded $diagnosis as factor \n")

# Show percentages
confirm <- prompt("\n Show percentages of diagnosis? (y/n): ")
if(confirm == "y") {
    print(round(prop.table(table(wbcd$diagnosis))*100, digits=2))
}

# Some numeric features hold way larger measures than others
cat("\n *** Normalization is required \n")
confirm <- prompt("\n Show summary of main features? (y/n): ")
if(confirm == "y") {
    cat("\n")
    print(summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")]))
}

# Create normalization function
normalize <- function(x) {
    return ((x-min(x)) / (max(x)-min(x)))
}

# Normalize numeric features
confirm <- prompt_norm("\n Select 0-1 standard normalization or z-scale standardization (n/z): ")
if(confirm == "n") {
    # Disperse numeric features now range from 0 to 1
    wbcd_norm <- as.data.frame(lapply(wbcd[2:ncol(wbcd)], normalize))
} else if(confirm == "z") {
    # The rescaled numeric features have now a mean zero
    wbcd_norm <- as.data.frame(scale(wbcd[-1]))
}

# Show rescaled features
feature_name <- prompt_str("\n Show summary of rescaled feature (whitespace to skip or name of the feature): ")
if(is.character(feature_name)) {
    print(summary(wbcd_norm[ , feature_name])) 
}

# Split dataframe for training(20%) and model testing(20%)
cat("\n *** Split dataframe for training and model testing \n")
train_rows <- round(nrow(wbcd_norm) * 80 / 100)
cat("\n *** Train rows " + train_rows + " \n")
# No need to randomize
wbcd_train <- wbcd_norm[1:train_rows, ]
cat("\n *** Test rows " + (nrow(wbcd_norm)-train_rows) + " \n")
wbcd_test <- wbcd_norm[(train_rows+1):(nrow(wbcd_norm)), ]
# Extracting diagnosis factor for perfomance comparison
cat("\n *** Extracting diagnosis factor for perfomance comparison \n")
wbcd_train_labels <- wbcd[1:train_rows, 1]
wbcd_test_labels <- wbcd[(train_rows+1):(nrow(wbcd_norm)), 1]

# Find an odd K value
k_val <- round(sqrt(train_rows))
is_k_odd <- (k_val%%2) != 0

wbcd_pred <- knn(train=wbcd_train, test=wbcd_test, cl=wbcd_train_labels, k=k_val)

# 2 false negatives, 0 false positives :S
CrossTable(x=wbcd_test_labels, y=wbcd_pred, prop.chisq=FALSE)
