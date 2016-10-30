
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

pause_enable <- prompt_ys("*** Enable pause (y/n): ")

# Replace patient id with serial integers
wbcd$id <- seq(1, nrow(wbcd), by=1)

# Diagnosis by mean/se/worst values

png("diagnosis_by_area_mean.png", width = 1000, height = 1000, res = 72)
wbcd$color <- ""
wbcd$color[wbcd$diagnosis=="M"] <- "red"
wbcd$color[wbcd$diagnosis=="B"] <- "green"
plot(wbcd$id, wbcd$area_mean,                              
     col = wbcd$color,  
     pch = 15,
     cex = .5,                                                
     xlab = "Patients",                                              
     ylab = "Area mean",                                   
     main = "Masses: area mean")
legend (x = 0, y = 2600, 
        legend = c("Benign", "Malignant"), 
        col = c("green", "red"), 
        pch = 15)
dev.off()
pause()


