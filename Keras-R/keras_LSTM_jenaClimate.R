library(readr)
library(ggplot2)
library(magrittr)
library(keras)

setwd("C:/Users/Casa/Desktop")

data_dir <- "./"
filename <- file.path(data_dir, "jena_climate_2009_2016.csv")
csv <- read_csv(filename)
float_data <- data.matrix(csv[,-1])

message("Plotting temperature...")
temp_plot <- ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line(color = 'blue')
print(temp_plot)

message("Center and scale...")
train_data <- float_data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(float_data, center = mean, scale = std)

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

generator <- function(data, lookback, delay, min_index, max_index, shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index)) max_index <- nrow(data) - delay - 1
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
    x <- array(0, dim = c(length(rows), lookback/step, n_columns))
    y <- array(0, dim = c(length(rows)))
    n_samples <- dim(x)[[2]]
    for (i in 1:length(rows)) {
      indices <- seq(rows[[i]] - lookback, rows[[i]], length.out = n_samples)
      x[i,,] <- data[indices,]
      y[[i]] <- data[rows[[i]] + delay,2]
    }
    list(x, y)
  }
}

train_gen <- generator(data, lookback = lookback, delay = delay, min_index = 1, max_index = 200000, shuffle = FALSE, step = step, batch_size = batch_size)
val_gen = generator(data, lookback = lookback, delay = delay, min_index = 200001, max_index = 300000, step = step, batch_size = batch_size)
test_gen <- generator(data, lookback = lookback, delay = delay, min_index = 300001, max_index = NULL, step = step, batch_size = batch_size)

message("Compiling model...")
model <- keras_model_sequential() %>%
         layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5, return_sequences = TRUE, input_shape = list(NULL, dim(data)[[-1]])) %>%
         layer_gru(units = 64, activation = "relu", dropout = 0.1, recurrent_dropout = 0.5) %>%
         layer_dense(units = 1)
model %>% compile(optimizer = optimizer_rmsprop(), loss = "mae")

message("Training...")
val_steps <- (300000 - 200001 - lookback)/batch_size
history <- model %>% fit_generator(train_gen, steps_per_epoch = 500, epochs = 40, validation_data = val_gen, validation_steps = val_steps)
plot(history)
