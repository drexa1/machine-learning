# DOC at https://keras.rstudio.com/index.html

library(readr)
library(ggplot2)
library(magrittr)
library(yaml)
library(keras)

setwd("C:/Users/drexa/git/machine-learning/Keras-R")

data_dir <- "./"
filename <- file.path(data_dir, "jena_climate_2009_2016.csv")
csv_data <- read_csv(filename)
float_data <- data.matrix(csv_data[,-1])

message("Plotting temperature...")
temp_plot <- ggplot(csv_data, aes(x = 1:nrow(csv_data), y = `T (degC)`)) + geom_line(color = 'blue')
print(temp_plot)

message("Center and scale...")
train_data <- float_data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(float_data, center = mean, scale = std)

lookback <- 1440
step <- 6
lookahead <- 144
batch_size <- 128

generator <- function(data, lookback, lookahead, min_index, max_index, shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index)) max_index <- nrow(data) - lookahead - 1
  index <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (index + batch_size >= max_index)
        index <<- min_index + lookback
      rows <- c(index:min(index+batch_size, max_index))
      index <<- index + length(rows)
    }
    n_columns <- dim(data)[[2]]
    x <- array(0, dim = c(length(rows), lookback/step, n_columns)) # [batchsize, 240, 14]
    y <- array(0, dim = c(length(rows)))
    n_samples <- dim(x)[[2]]
    for (i in 1:length(rows)) {
      indices <- seq(rows[[i]] - lookback, rows[[i]], length.out = n_samples)
      x[i,,] <- data[indices,]
      y[[i]] <- data[rows[[i]] + lookahead,2] # [lookahead row, T (degC)]
    }
    list(x, y)
  }
}

model <- keras_model_sequential() %>%
    layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5, return_sequences = TRUE, input_shape = list(NULL, dim(data)[[-1]])) %>%
    layer_gru(units = 64, activation = "relu", dropout = 0.1, recurrent_dropout = 0.5) %>%
    layer_dense(units = 1)

message("Compiling model...")
model %>% compile(optimizer = optimizer_rmsprop(), loss = "mae")

train_gen <- generator(data, lookback = lookback, lookahead = lookahead, min_index = 1, max_index = 200000, shuffle = FALSE, step = 1, batch_size = batch_size)
val_gen = generator(data, lookback = lookback, lookahead = lookahead, min_index = 200001, max_index = 300000, step = 1, batch_size = batch_size)
test_gen <- generator(data, lookback = lookback, lookahead = lookahead, min_index = 300001, max_index = NULL, step = 1, batch_size = batch_size)

model_callbacks <- function() {
  callback_model_checkpoint <- callback_model_checkpoint("jenaClimate_checkpoints.h5")
  callback_early_stopping <- callback_early_stopping(monitor="acc", patience=1)
  callback_reduce_lr_on_plateau <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)
  callback_tensorboard <- callback_tensorboard(log_dir="tensorboard_jenaClimate")
  list(callback_model_checkpoint, callback_early_stopping, callback_reduce_lr_on_plateau, callback_tensorboard)
}

message("Training...")
val_steps <- (300000 - 200001 - lookback)/batch_size
model_callbacks <- model_callbacks()
history <- model %>% fit_generator(train_gen, 
                                   steps_per_epoch = 500, 
                                   epochs = 40,
                                   callbacks = model_callbacks,
                                   validation_data = val_gen, 
                                   validation_steps = val_steps
                                   )

save_model_weights_hdf5(model, "jenaClimate_weights.h5", overwrite = TRUE)

plot(history)
