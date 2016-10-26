library("class")
library("gmodels")

# Import data
wbcd <- read.csv("datasets/wisc_bc_data.csv", stringsAsFactors = FALSE)

# Exclude ID to avoid spurius predictions
id_idx <- match("id", names(wbcd))
wbcd <- wbcd[-id_idx]
# Recode diagnosis as a factor
table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis, c("B", "M"), c("Benign", "Malignant"))
# Show percentages
round(prop.table(table(wbcd$diagnosis))*100, digits=2)

# Some numeric features hold way larger measures than others
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])
# Z-scale standardization to allow outliers to weight heavy
z_scale_standardize <- function(dataframe) {
    return (as.data.frame(scale(dataframe[])))
}
# Normalize numeric features
wbcd_norm <- z_scale_standardize(wbcd)
# Disperse features should have a 0 mean
# Presence further than -3 ... 3 indicate heavy outliers
summary(wbcd_norm$area_mean)

# Split dataframe for training(20%) and model testing(20%)
train_rows <- round(nrow(wbcd_norm) * 80 / 100)
# No need to randomize
wbcd_train <- wbcd_norm[1:train_rows, ]
wbcd_test <- wbcd_norm[(train_rows+1):(nrow(wbcd_norm)), ]
# Extract labels for diagnosis factor
wbcd_train_labels <- wbcd[1:train_rows, 1]
wbcd_test_labels <- wbcd[(train_rows+1):(nrow(wbcd_norm)), 1]

# Find an odd K value
k_val <- round(sqrt(train_rows))
is_k_odd <- (k_val%%2) != 0

wbcd_pred <- knn(train=wbcd_train, test=wbcd_test, cl=wbcd_train_labels, k=k_val)

# 2 false negatives, 0 false positives :S
CrossTable(x=wbcd_test_labels, y=wbcd_pred, prop.chisq=FALSE)
