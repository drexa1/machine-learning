library(oro.dicom)
library(oro.nifti)

options(warn = -1)
set.seed(123)
setwd("C:/Users/drexa/git/MachineLearning/Data-science-bowl-2017/sample_images")

IMAGES_PATH = paste(getwd(), .Platform$file.sep, '00cba091fa4ad62cc3200a657aeb957e', sep = '')

dcmImages <- readDICOM(IMAGES_PATH, recursive = FALSE, exclude = NULL, verbose = TRUE)
dcm.info <- dicomTable(dcmImages$hdr)
print(names(dcm.info)) # Header columns
# unique(dcm.info["0020-1041-SliceLocation"])

# image(t(patient_images$img[[18]]), col = grey(0:64 / 64), axes = FALSE)
for(i in 1:length(dcmImages$img)) {
    dcm.stack <- list(hdr = dcmImages$hdr[i], img = dcmImages$img[i])
    dcm.nifti <- dicom2nifti(dcm.stack, DIM = 3, descrip = c("a", "b"))
    print(dcm.nifti)
    writeNIfTI(dcm.nifti, paste0("NIfTI/", i))
}