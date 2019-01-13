## AUTHOR: Sanandeesh Kamat
## DATE:   November 2018
################
## DESCRIPTION: 
## In this problem, we apply Keras/Tensorflow Neural Networks to predict the price a wine sold for based on select words from the corresponding wine review.
################

library(keras)
library(randomForest)
# use_condaenv("r-tensorflow") # My computer needs this!

wines = read.csv("wines.csv")

words = c("sweet", "acid", "earthy", "fruit", "tannin", "herb",
          "tart", "spice", "smooth", "full", "intense",
          "wood", "soft", "dry", "apple", "pear", "cherry",
          "berry", "aroma", "citrus", "lemon", "lime", "peach", "blossom",
          "sugar", "simple", "cinnamon", "ripe",
          "crisp", "honey", "brisk", "fresh", "sour", "floral", 
          "dark", "complex", "oak", "balance", "caramel", "plum", "mint",
          "apricot", "cream", "vanilla", "butter", "sharp")

## NOTE: wines[, "apple"] is either 0/1. wines[i, "apple"] == 1 if the review of wine i uses 
##       the word "apple" and 0 else.

wines = wines[wines$price < 300, ]

X = wines[, words]  # [nxp] Input Feature Matrix
X = as.matrix(X)
X = scale(X)

Y = wines[, "price"] # [nx1] Response Vector \in [0 ... 300]

ntrain = 200 # Modified Model 3
# ntrain = 2000 # Original 
test_ix = sample(nrow(wines), nrow(wines) - ntrain) # Random Test Indeces

x_train = X[-test_ix, ]
y_train = Y[-test_ix]

x_test = X[test_ix, ]
y_test = Y[test_ix]

## Neural Network
model <- keras_model_sequential() 
## =========== FILL IN: add layers here =========== 
# ======== Original Model ======== 
if (0) {
  model %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'linear')
}
# ======== Modified Model 1 ======== 
# Add 5 additional layers. Changed the widths to be [64, 128, 256, 128, 64, 32, 16]
if (0) {
  model %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 256, activation = 'relu') %>% # 1
  layer_dense(units = 128, activation = 'relu') %>% # 2
  layer_dense(units = 64, activation = 'relu') %>%  # 3
  layer_dense(units = 32, activation = 'relu') %>%  # 4
  layer_dense(units = 16, activation = 'relu') %>%  # 5
  layer_dense(units = 1, activation = 'linear')
}
# ======== Modified Model 3 ======== 
if (1) {
  model %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.9) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.9) %>%
  layer_dense(units = 1, activation = 'linear')
}

model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(),
  metrics = c('mse')
)

model %>% fit(
  x_train, y_train, 
  epochs = 20, batch_size = 32 # Original
  # epochs = 1, batch_size = 64 # Modified Model 2
)

y_nn = model %>% predict(x_test)
err_nn = sqrt(mean((y_nn - y_test)^2))

## Random Forest
library(randomForest)

rf_res = randomForest(x=x_train, y=y_train)
y_rf = predict(rf_res, newdata=x_test)

err_rf = sqrt(mean((y_rf - y_test)^2))

## Linear Model
lm_res = lm.fit(x=cbind(x_train, 1), y=y_train)
y_lm = cbind(x_test, 1) %*% lm_res$coefficients 

err_lm = sqrt(mean((y_lm - y_test)^2))


err_base = sqrt(mean((y_test - mean(y_train))^2))
print(sprintf("Baseline Err: %.3f  Neural Net Err: %.3f  Linear Err: %.3f  RF: %.3f", 
              err_base, err_nn, err_lm, err_rf))
