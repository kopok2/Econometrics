# packages
library("LaplacesDemon")
library("tsoutliers")

# constants
n_bootstrap = 40
data_len = 50
k = 3

# generate data
b <- 300
x1 <- rnorm(data_len, 14, 23)
x2 <- rnorm(data_len, -20, 34)
x3 <- rnorm(data_len, 70, 12)
a1 <- 3
a2 <- -2
a3 <- 0
noise <- rnorm(data_len, 50, 30)

y <- b + a1 * x1 + a2 * x2 + a3 * x3 + noise
data <- data.frame(y, x1, x2, x3)

# fit base
model <- lm(y ~ x1 + x2 + x3 , data = data)
params <- model$coefficients
y_hat <- params["(Intercept)"] + params["x1"] * x1 + params["x2"] * x2 + params["x3"] * x3
e <- resid(model)
y_mean_vec <- rep(mean(y), data_len)
r2 <- summary(model)$r.squared
summary(model)

# 1. sample residuals
print("Losowanie reszt")
sample_1_a0 <- 1:n_bootstrap
sample_1_a1 <- 1:n_bootstrap
sample_1_a2 <- 1:n_bootstrap
sample_1_a3 <- 1:n_bootstrap
sample_1_r2 <- 1:n_bootstrap

for(i in 1:n_bootstrap)
{
  e_bootstrap <- sample(e, data_len, replace = TRUE)
  y_star_1 <- y_hat + e_bootstrap
  data_1 <- data.frame(y_star_1, x1, x2, x3)
  model_1 <- lm(y_star_1 ~ x1 + x2 + x3, data = data_1)
  params_1 <- model_1$coefficients
  sample_1_a0[i] <- params_1["(Intercept)"]
  sample_1_a1[i] <- params_1["x1"]
  sample_1_a2[i] <- params_1["x2"]
  sample_1_a3[i] <- params_1["x3"]
  sample_1_r2[i] <- summary(model_1)$r.squared
}

print("Przedzialy ufnosci empiryczne:")
p.interval(sample_1_a0)
p.interval(sample_1_a1)
p.interval(sample_1_a2)
p.interval(sample_1_a3)
p.interval(sample_1_r2)

print("Przedzialy ufnosci wg teorii:")
t.test(sample_1_a0)$conf.int
t.test(sample_1_a1)$conf.int
t.test(sample_1_a2)$conf.int
t.test(sample_1_a3)$conf.int

hist(sample_1_a0, probability = TRUE)
lines(density(sample_1_a0), lwd=4, col = 'red')
hist(sample_1_a1, probability = TRUE)
lines(density(sample_1_a1), lwd=4, col = 'red')
hist(sample_1_a2, probability = TRUE)
lines(density(sample_1_a2), lwd=4, col = 'red')
hist(sample_1_a3, probability = TRUE)
lines(density(sample_1_a3), lwd=4, col = 'red')
hist(sample_1_r2, probability = TRUE)
lines(density(sample_1_r2), lwd=4, col = 'red')

# 2. sample pairs
print("Losowanie par (y, x)")
sample_2_a0 <- 1:n_bootstrap
sample_2_a1 <- 1:n_bootstrap
sample_2_a2 <- 1:n_bootstrap
sample_2_a3 <- 1:n_bootstrap
sample_2_r2 <- 1:n_bootstrap

for(i in 1:n_bootstrap)
{
  data_2 <- data[sample(nrow(data), data_len, replace = TRUE), ]
  model_2 <- lm(y ~ x1 + x2 + x3, data = data_2)
  params_2 <- model_2$coefficients
  sample_2_a0[i] <- params_2["(Intercept)"]
  sample_2_a1[i] <- params_2["x1"]
  sample_2_a2[i] <- params_2["x2"]
  sample_2_a3[i] <- params_2["x3"]
  sample_2_r2[i] <- summary(model_2)$r.squared
}

print("Przedzialy ufnosci empiryczne:")
p.interval(sample_2_a0)
p.interval(sample_2_a1)
p.interval(sample_2_a2)
p.interval(sample_2_a3)
p.interval(sample_2_r2)

print("Przedzialy ufnosci wg teorii:")
t.test(sample_2_a0)$conf.int
t.test(sample_2_a1)$conf.int
t.test(sample_2_a2)$conf.int
t.test(sample_2_a3)$conf.int

hist(sample_2_a0, probability = TRUE)
lines(density(sample_2_a0), lwd=4, col = 'red')
hist(sample_2_a1, probability = TRUE)
lines(density(sample_2_a1), lwd=4, col = 'red')
hist(sample_2_a2, probability = TRUE)
lines(density(sample_2_a2), lwd=4, col = 'red')
hist(sample_2_a3, probability = TRUE)
lines(density(sample_2_a3), lwd=4, col = 'red')
hist(sample_2_r2, probability = TRUE)
lines(density(sample_2_r2), lwd=4, col = 'red')

# other
summary(e)
hist(e, probability = TRUE)
lines(density(e), lwd=4, col = 'red')
jarque.bera.test
