## AUTHOR: Sanandeesh Kamat
## DATE:   November 2018
################
## DESCRIPTION: 
## In this problem, we Keras/Tensorflow Neural Networks to classify whether or not a county voted for Trump or Clinton based on an array of socio-demographic features.
################


library(keras)
# use_condaenv("r-tensorflow") # My computer needs this!

votes = read.csv("votes.csv")
votes$prefer_trump = votes$trump > votes$clinton
features = c("white", "black", "poverty", "density", "bachelor", "highschool", "age65plus",
             "income", "age18under", "population2014")

X = votes[, features]       # [nxp] Input Feature Matrix
X = scale(X)
Y = votes[, "prefer_trump"] # [nx1] Binary Response Vector 

# ntrain = 600 # Original
ntrain = 200 # Modified Model 3
test_ix = sample(nrow(votes), nrow(votes) - ntrain) # Random Test Sample Indeces

Y[Y==0] = 0

x_train = X[-test_ix, ] # Training Input Feature Matrix
Y1      = Y[-test_ix]   # Training Binary Response Vector

x_test = X[test_ix, ]   # Test Input Feature Matrix
Y2     = Y[test_ix]     # Test Binary Response Vector

## Fit neural network
y_train <- to_categorical(Y1, 2)
y_test <- to_categorical(Y2, 2)

# DEFINE A MODEL: Keras Model composed of a linear stack of layers
model <- keras_model_sequential()

# CORE LAYERS:
# layer_dense():   Add a densely-connected NN layer to an output
# layer_dropout(): Applies Dropout to the input 
# ================================ Original Model ===========================
if (0) {
model %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 2, activation = 'softmax')
}
# ================================ Modified Model 1 ===========================
# Add 5 additional layers. Changed the widths to be [128, 128, 128, 128, 64, 32, 16]
if (0) {
model %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>% # 1st added
  layer_dense(units = 128, activation = 'relu') %>% # 2nd added
  layer_dense(units = 64, activation = 'relu') %>% # 3th added
  layer_dense(units = 32, activation = 'relu') %>% # 4th added
  layer_dense(units = 16, activation = 'relu') %>% # 5th added
  layer_dense(units = 2, activation = 'softmax')
}
# ================================ Modified Model 3 ===========================
# Set dropout rates to 0. Reduced num training samples to 300. (from 600)
if (1) {
model %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0) %>%
  layer_dense(units = 2, activation = 'softmax')
}




# COMPILE: Configure a Keras model for training
# compile(object, optimizer, loss, metrics = NULL)
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fit(object, x = NULL, y = NULL, batch_size = NULL, epochs = 10, verbose = 1, callbacks = NULL, ...)
# Train a Keras model for a fixed number of epochs (iterations)
model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 32 # Original
  # epochs = 15, batch_size = 1 # Modified Model 2
)

y_hat = model %>% predict_classes(x_test)
err_nn = mean(abs(y_hat - Y2))


## Logistic regression
X1 = cbind(x_train, rep(1, nrow(x_train)))
X2 = cbind(x_test, rep(1, nrow(x_test)))
X2 = as.matrix(X2)

w_lr = glm.fit(X1, Y1, family=binomial(link="logit"))$coefficients

## ============= Make logistic regression prediction ============= 
Y2_lr = ((X2%*%w_lr) >= 0)

err_lr = mean(abs(Y2_lr - Y2))


err_base = ifelse(mean(Y1) > 0.5, mean(1 - Y2), mean(Y2))
print(sprintf("Baseline Error: %.3f  Neural Net Error: %.3f  LR Error: %.3f", 
              err_base,
              err_nn, err_lr))