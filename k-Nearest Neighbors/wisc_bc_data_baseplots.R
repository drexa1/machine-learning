
# Import data
setwd("C:/Users/DR186049/git/MachineLearning/k-Nearest Neighbors")
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

# Layout
color <- ""
# par(mar=rep(1,4))
par(mfrow=c(3,4))

features <- list()
for(feature in selected_features) {
    cat(paste("*** Generating plot for ",feature," \n",sep=""))
  
    plot_name <- paste("diagnosis_by_",selected_features,sep="")
    # Render to file
    # png(paste(plot_name,".png",sep=""), width=1000, height=1000, res=72)
    
    # features[n][1]: id, 
    # features[n][2]: feature value, 
    # features[n][3]: diagnosis
    features[[length(features)+1]] <- data.frame(wbcd$id, wbcd[ , feature], wbcd$diagnosis)
    
    color[features[[length(features)]][3]=="M"] <- "red"
    color[features[[length(features)]][3]=="B"] <- "black"
    
    # REF http://rstudio-pubs-static.s3.amazonaws.com/7953_4e3efd5b9415444ca065b1167862c349.html
    plot(data.frame(features[[length(features)]][1], features[[length(features)]][2]),                              
         col = color,  
         pch = 20,
         cex = .5,                                                
         xlab = "Patients",                                              
         ylab = feature,                                   
         main = paste("Masses: ",gsub(paste("_",feature_type,sep=""),"",feature),sep=""))
    # legend (x=0, y=max(features[[length(features)]][2]), 
            # legend = c("Benign", "Malignant"), 
            # col = c("black", "red"), 
            # pch = 15)
}
