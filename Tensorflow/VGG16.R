library("tensorflow")
library("magrittr")
library("jpeg")
library("readr")
library("grid")
library("ggplot2")

options(warn = -1)
set.seed(123)

cat("\n")
cat("------------------------------------------------ \n")
cat(" Classification using Convoluted neural networks \n")
cat("------------------------------------------------ \n")

setwd("C:/Users/drexa/git/MachineLearning/Tensorflow")
# setwd("C:/Users/dr186049/git/MachineLearning/Tensorflow")

# Import image
img1 <- readJPEG('./image.jpg')
size = dim(img1)
img_array = array(255 * img1, c(1, size[1], size[2], size[3]))

slim = tf$contrib$slim
# Build the model from scratch
tf$reset_default_graph()

# Tensor placeholder
images = tf$placeholder(tf$float32, shape(NULL, NULL, NULL, 3))
images_scaled = tf$image$resize_images(images, shape(224, 224))
cat("*** Image imported and scaled \n")

# Definition of VGG16 network
cat("*** Generating convnet model... \n")
vgg_16 = slim$conv2d(images_scaled, 64, shape(3, 3), scope = 'vgg_16/conv1/conv1_1') %>%
         slim$conv2d(64, shape(3, 3), scope = 'vgg_16/conv1/conv1_2') %>%
         slim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool1') %>%

         slim$conv2d(128, shape(3, 3), scope = 'vgg_16/conv2/conv2_1') %>%
         slim$conv2d(128, shape(3, 3), scope = 'vgg_16/conv2/conv2_2') %>%
         slim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool2') %>%

         slim$conv2d(256, shape(3, 3), scope = 'vgg_16/conv3/conv3_1') %>%
         slim$conv2d(256, shape(3, 3), scope = 'vgg_16/conv3/conv3_2') %>%
         slim$conv2d(256, shape(3, 3), scope = 'vgg_16/conv3/conv3_3') %>%
         slim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool3') %>%

         slim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv4/conv4_1') %>%
         slim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv4/conv4_2') %>%
         slim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv4/conv4_3') %>%
         slim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool4') %>%

         slim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv5/conv5_1') %>%
         slim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv5/conv5_2') %>%
         slim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv5/conv5_3') %>%
         slim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool5') %>%

         slim$conv2d(4096, shape(7, 7), padding = 'VALID', scope = 'vgg_16/fc6') %>%
         slim$conv2d(4096, shape(1, 1), scope = 'vgg_16/fc7') %>%

         # Setting the activation_fn=NULL does not work, so we get a ReLU
slim$conv2d(1000, shape(1, 1), scope = 'vgg_16/fc8') %>%
         tf$squeeze(shape(1, 2), name = 'vgg_16/fc8/squeezed')

# Tensorboard graph
cat("*** Model rendered to tensorboard \n")
tf$summary$FileWriter('./vgg16', tf$get_default_graph())$close()

# Restore weights
cat("*** Restoring weights\n")
restorer = tf$train$Saver()
session = tf$Session()
# tf$global_variables_initializer()
restorer$restore(session, './vgg_16.ckpt')

# Feed the placeholder
cat("*** Feeding placeholder... \n")
vgg_16_vals = session$run(vgg_16, dict(images = img_array))

# Sort the propabilities
probs = exp(vgg_16_vals) / sum(exp(vgg_16_vals))
idx = sort.int(vgg_16_vals, index.return = TRUE, decreasing = TRUE)$ix[1:5]

# Read class labels
labels = read_delim("labels.txt", "\t", escape_double = FALSE, trim_ws = TRUE, col_names = FALSE)

# Plot
g = rasterGrob(img1, interpolate = TRUE)
text = ""
for (i in idx) {
    text = paste0(text, labels[i,][[1]], " ", round(probs[i], 2), "\n")
}
p = print(ggplot(data.frame(d = 1:3)) +
          annotation_custom(g) +
          annotate('text', x = 0.05, y = 0.05, label = text, size = 5, hjust = 0, vjust = 0, color = 'red') +
          xlim(0, 1) +
          ylim(0, 1))
