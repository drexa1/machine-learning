
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

# Replace patient id with serial integers
wbcd$id <- seq(1, nrow(wbcd), by=1)

# Select mean|se|worst features
feature_type <- prompt_feature("*** Select feature to plot [mean|se|worst]: ")
selected_features <- names(wbcd)[grep(paste("_",feature_type,sep=""), names(wbcd))]
