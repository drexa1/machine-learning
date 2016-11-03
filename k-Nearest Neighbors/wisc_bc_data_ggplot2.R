library("ggplot2")

# Import data
setwd("C:/Users/drexa/git/R/MachineLearning/k-Nearest Neighbors")
wbcd <- read.csv("../datasets/wisc_bc_data.csv", stringsAsFactors = TRUE)
cat("\n *** Wisconsin General Hospital breast cancer dataset imported \n")

prompt_feature <- function(str) {
    input <- readline(str)
    input <- ifelse(grepl("mean|se|worst", input), as.character(input), -1)
    if(!is.character(input)) {
        prompt_feature(str)
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

# Replace patient id with serial integers
wbcd$id <- seq(1, nrow(wbcd), by=1)

# Select mean|se|worst features
feature_type <- prompt_feature("*** Select feature to plot [mean|se|worst]: ")
selected_features <- names(wbcd)[grep(paste("_",feature_type,sep=""), names(wbcd))]

# Subset for the selected features
wbcd_subset <- wbcd[ ,selected_features]

# Create normalization function
normalize <- function(x) {
    return ((x-min(x)) / (max(x)-min(x)))
}
# Normalize numeric features
confirm <- prompt_norm("*** Select 0-1 standard normalization or z-scale standardization (n/z): ")
if(confirm == "n") {
    wbcd_norm <- as.data.frame(lapply(wbcd_subset, normalize))
} else if(confirm == "z") {
    wbcd_norm <- as.data.frame(scale(wbcd_subset))
}

# Reattach dianosis factor
# wbcd_norm$diagnosis <- wbcd$diagnosis

# wbcd_trans <- data.frame(t(wbcd_norm))
plot <- ggplot(wbcd_norm) + coord_flip()
print(plot)





