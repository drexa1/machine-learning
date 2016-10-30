
# Import data
setwd("C:/Users/drexa/git/R/Machine Learning/k-Nearest Neighbors")
wbcd <- read.csv("../datasets/wisc_bc_data.csv", stringsAsFactors = TRUE)
cat("*** Wisconsin General Hospital breast cancer dataset imported \n")

pause_enable <<- "y"
pause <- function() {
    if(pause_enable=="y") {
        invisible((prompt="Press [enter] to continue \n"))  
    }
}
prompt_ys <- function(str) {
    input <- readline(str)
    input <- ifelse(grepl("y|n", input), as.character(input), -1)
    if(!is.character(input)) {
        prompt_norm(str)
    }
    return (as.character(input))
}
prompt_feature <- function(str) {
    input <- readline(str)
    input <- ifelse(grepl("mean|se|worst", input), as.character(input), -1)
    if(!is.character(input)) {
        prompt_feature(str)
    }
    return (as.character(input))
}

pause_enable <- prompt_ys("*** Enable pause (y/n): ")

# Replace patient id with serial integers
wbcd$id <- seq(1, nrow(wbcd), by=1)

# Diagnosis by mean/se/worst values
plots <- list()
feature_type <- prompt_feature("*** Select feature to plot [mean|se|worst]: ")
selected_features <- names(wbcd)[grep(paste("_",feature_type,sep=""), names(wbcd))]
for(feature_name in selected_features) {
    cat(paste("*** Generating plot for ",feature_name," \n",sep=""))
    plot_name <- paste("diagnosis_by_",feature_name,sep="")
    png(paste(plot_name,".png",sep=""), width = 1000, height = 1000, res = 72)
    wbcd$color <- ""
    wbcd$color[wbcd$diagnosis=="M"] <- "red"
    wbcd$color[wbcd$diagnosis=="B"] <- "green"
    plots[[length(plots)+1]] <- plot(wbcd$id, wbcd_norm[ , feature_name],                              
         col = wbcd$color,  
         pch = 15,
         cex = .5,                                                
         xlab = "Patients",                                              
         ylab = feature_name,                                   
         main = paste("Masses: ",feature_name,sep=""))
    legend (x = 0, y = max(wbcd_norm[ , feature_name]), 
            legend = c("Benign", "Malignant"), 
            col = c("green", "red"), 
            pch = 15)
    dev.off()
    pause()
}

