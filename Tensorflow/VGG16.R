library("tensorflow") # Tensorflow
library("magrittr") # Pipes
library("ggplot2") # Graphics plotting

options(warn = -1)
set.seed(123) # Experiment reproducibility

cat("\n")
cat("------------------------------------------------ \n")
cat(" Classification using convoluted network \n")
cat("------------------------------------------------ \n")

# setwd("C:/Users/dr186049/git/MachineLearning/Tensorflow")
setwd("C:/Users/drexa/git/MachineLearning/Tensorflow")

# Import image
img <- jpeg::readJPEG('./image.jpg')
size = dim(img)
img_array = array(255 * img, c(1, size[1], size[2], size[3]))
cat("*** Image read and rescaled \n")

# Initialize graph
cat("*** Initializing graph \n")
tf$reset_default_graph()
tfSlim = tf$contrib$slim

# Allocate placeholder
img_placeholder = tf$placeholder(tf$float32, shape(NULL, NULL, NULL, 3))
img_placeholder_scaled = tf$image$resize_images(img_placeholder, shape(224, 224))

# Define VGG16
vgg16 = tfSlim$conv2d(img_placeholder_scaled, 64, shape(3, 3), scope = 'vgg_16/conv1/conv1_1') %>%
        tfSlim$conv2d(64, shape(3, 3), scope = 'vgg_16/conv1/conv1_2') %>%
        tfSlim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool1') %>%
        # conv2
        tfSlim$conv2d(128, shape(3, 3), scope = 'vgg_16/conv2/conv2_1') %>%
        tfSlim$conv2d(128, shape(3, 3), scope = 'vgg_16/conv2/conv2_2') %>%
        tfSlim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool2') %>%
        # conv3
        tfSlim$conv2d(256, shape(3, 3), scope = 'vgg_16/conv3/conv3_1') %>%
        tfSlim$conv2d(256, shape(3, 3), scope = 'vgg_16/conv3/conv3_2') %>%
        tfSlim$conv2d(256, shape(3, 3), scope = 'vgg_16/conv3/conv3_3') %>%
        tfSlim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool3') %>%
        # conv4
        tfSlim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv4/conv4_1') %>%
        tfSlim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv4/conv4_2') %>%
        tfSlim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv4/conv4_3') %>%
        tfSlim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool4') %>%
        # conv5
        tfSlim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv5/conv5_1') %>%
        tfSlim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv5/conv5_2') %>%
        tfSlim$conv2d(512, shape(3, 3), scope = 'vgg_16/conv5/conv5_3') %>%
        tfSlim$max_pool2d(shape(2, 2), scope = 'vgg_16/pool5') %>%
        # fc1
        tfSlim$conv2d(4096, shape(7, 7), padding = 'VALID', scope = 'vgg_16/fc6') %>%
        # fc2
        tfSlim$conv2d(4096, shape(1, 1), scope = 'vgg_16/fc7') %>%
        # fc3
        tfSlim$conv2d(1000, shape(1, 1), scope = 'vgg_16/fc8') %>% # activation_fn=NULL does not work, we get a ReLU
        tf$squeeze(shape(1, 2), name = 'vgg_16/fc8/squeezed')

# Print to Tensorboard
tf$train$SummaryWriter('./vgg16', tf$get_default_graph())$close()

# Restoring pre - trained weights
cat("*** Restoring pre-trained weights \n")
session = tf$Session()
restorer = tf$train$Saver()
restorer$restore(session, './vgg_16.ckpt')

# Calculating probabilities
cat("*** Feeding data to the graph \n")
vgg16_values = session$run(vgg16, dict(img_placeholder = img_array))
probs = exp(vgg16_values) / sum(exp(vgg16_values))
idx = sort.int(vgg16_values, index.return = TRUE, decreasing = TRUE)$ix[1:5]

# Reading class labels
cat("*** Reading class labels \n")
labels = readr::read_delim("labels.txt", "\t", escape_double = FALSE, trim_ws = TRUE, col_names = FALSE)

# Plotting
cat("*** Plotting probabilities \n")
g = grid::rasterGrob(img, interpolate = TRUE)
text = ""
for (id in idx) {
    text = paste0(text, labels[id,][[1]], " ", round(probs[id], 3), "\n")
}
p = ggplot(data.frame(d = 1:3)) + 
           annotation_custom(g) +
           annotate('text', x = 0.05, y = 0.05, label = text, size = 4, hjust = 0, vjust = 0, color = 'darkgrey') + 
           xlim(0, 1) + ylim(0, 1)
           p %>% print
