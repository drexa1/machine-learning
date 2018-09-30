# Targets:
# Mill main drive - Main motor current
# Mill weight

library(readr)
library(ggplot2)
library(magrittr)
library(keras)

setwd("C:/Users/drexa/git/machine-learning/Keras-R")

data_dir <- "./"
filename <- file.path(data_dir, "sag_mill.csv")
csv_data <- read_csv(filename, na="")
dropna_csv_data <- na.omit(csv_data)
na_summary <- sapply(dropna_csv_data, function(x) sum(is.na(x)))
float_data <- data.matrix(dropna_csv_data[,-1])

message("Center and scale...")
data <- scale(float_data, center = TRUE, scale = TRUE)

generator <- function(data, lookback, lookahead, min_index, max_index, shuffle = FALSE, batch_size = 128, step = 1) {
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
    n_samples <- dim(x)[[2]]
    target_col <- 2
    x <- array(0, dim = c(length(rows), lookback/step, n_columns)) # [batchsize, 240, 14]
    y <- array(0, dim = c(length(rows)))
    for (i in 1:length(rows)) {
      indices <- seq(rows[[i]] - lookback, rows[[i]], length.out = n_samples)
      x[i,,] <- data[indices,]
      y[[i]] <- data[rows[[i]] + lookahead, target_col] # [lookahead row, target]
    }
    list(x, y)
  }
}

lookback <- 15
lookahead <- 10
batch_size <- 128

train_gen <- generator(data, lookback = lookback, lookahead = lookahead, min_index = 1, max_index = 300000, shuffle = FALSE)
val_gen = generator(data, lookback = lookback, lookahead = lookahead, min_index = 300001, max_index = 400000)
test_gen <- generator(data, lookback = lookback, lookahead = lookahead, min_index = 400001, max_index = NULL)

model <- keras_model_sequential() %>%
    layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5, return_sequences = TRUE, input_shape = list(NULL, dim(data)[[2]])) %>%
    layer_gru(units = 64, activation = "relu", dropout = 0.1, recurrent_dropout = 0.5) %>%
    layer_dense(units = 1)

message("Compiling model...")
model %>% compile(optimizer = optimizer_rmsprop(), loss = "mae")

model_callbacks <- function() {
    callback_model_checkpoint <- callback_model_checkpoint("sag_mill_0.h5")
    callback_early_stopping <- callback_early_stopping(monitor="acc", patience=1)
    callback_reduce_lr_on_plateau <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1)
    callback_tensorboard <- callback_tensorboard(log_dir="tensorboard_sag_mill_0")
    list(callback_model_checkpoint, callback_early_stopping, callback_reduce_lr_on_plateau, callback_tensorboard)
}

message("Training...")
val_steps <- (400000 - 300001 - lookback)/batch_size
model_callbacks <- model_callbacks()
history <- model %>% fit_generator(train_gen, 
                                   steps_per_epoch = 500, 
                                   epochs = 40,
                                   callbacks = model_callbacks,
                                   validation_data = val_gen, 
                                   validation_steps = val_steps
                                   )

save_model_weights_hdf5(model, "sag_mill_0.h5", overwrite = TRUE)

plot(history)
