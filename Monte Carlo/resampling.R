
# Find some complex distribution
# dist = sample(1:100, size = 10000, replace = TRUE)
attach(faithful)
dist = eruptions

summary(dist)
hist(dist)
hist(dist, seq(1.6, 5.2, 0.2), prob=TRUE)
lines(density(dist, bw=0.1))
rug(dist)

original_mean = mean(dist)

# do this a million times
sample_means = integer()
for (i in 1:100000) {
    # subsample batch of 5 elements from the original distribution and save them in a vector
    resamp_i = sample(dist, size = 5, replace = TRUE)
    mean_i = mean(resamp_i)
    sample_means = c(sample_means, mean_i)
}

# is the mean of subsamples similar to the mean of the original distribution?
mean(sample_means)

