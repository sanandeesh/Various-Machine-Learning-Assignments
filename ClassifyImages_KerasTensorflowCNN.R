## AUTHOR: Sanandeesh Kamat
## DATE:   November 2018
################
## DESCRIPTION: 
## In this problem, we use Keras/Tensorflow Convolutional Neural Networks to classify the type of clothing based on an image of the clothing.
## The standard data set comes from 'dataset_fashion_mnist()'.
################

library(keras)
# use_condaenv("r-tensorflow") # My computer needs this

mydata <- dataset_fashion_mnist()

x_train <- mydata$train$x
y_train <- mydata$train$y
x_test <- mydata$test$x
y_test <- mydata$test$y

x_train <- x_train / 255
x_test <- x_test / 255

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


## First fit a vanilla multiclass neural net
model1 <- keras_model_sequential() 
model1 %>% layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model1 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

model1 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 32, 
  validation_split = 0.2
)

y_hat = model1 %>% predict_classes(x_test)
Y2 = mydata$test$y
err1 = mean((y_hat - Y2 != 0))


## Try convolutional neural net
model_cnn <- keras_model_sequential() 

# ================================ Original Model ===========================
if (0) {
model_cnn %>%
  layer_conv_2d(filters=32, kernel_size=c(3,3), padding="same", stride=1,
                input_shape=c(28, 28, 1), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')
}
# ================================ Modified Model 1 ===========================
# Added 5 additional dense layers, and set the widths to [128, 256, 128, 64, 32, 16], respectively.
if (0) {
model_cnn %>%
  layer_conv_2d(filters=32, kernel_size=c(3,3), padding="same", stride=1,
                input_shape=c(28, 28, 1), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 256, activation = 'relu') %>% # 1st added
  layer_dense(units = 128, activation = 'relu') %>% # 2nd added
  layer_dense(units = 64,  activation = 'relu') %>% # 3rd added
  layer_dense(units = 32,  activation = 'relu') %>% # 4th added
  layer_dense(units = 16,  activation = 'relu') %>% # 5th added
  layer_dense(units = 10,  activation = 'softmax')
}

# ================================ Modified Model 2 ===========================
# Change the dropout rate and add additional convolutional layers and pooling layers.
if (1) {
model_cnn %>%
  layer_conv_2d(filters=32, kernel_size=c(7,7), padding="same", stride=1,
                input_shape=c(28, 28, 1), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=32, kernel_size=c(5,5), padding="same", stride=1, # 1st added convolutional layer 
                input_shape=c(28, 28, 1), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%                              # 1st added pooling layer
  layer_conv_2d(filters=32, kernel_size=c(3,3), padding="same", stride=1, # 2nd added convolutional layer
                input_shape=c(28, 28, 1), activation="relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%                              # 2nd added pooling layer
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
}

model_cnn %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

model_cnn %>% fit(
  x_train, y_train,
  # epochs = 3, batch_size = 128, # Modified Model 2 
  epochs = 10, batch_size = 32, # Original
  validation_split = 0.2
)

y_hat = model_cnn %>% predict_classes(x_test)
Y2 = mydata$test$y

err_cnn = mean((y_hat - Y2 != 0))

table(Y2)
err_base = 0.9

print(sprintf("Baseline Error: %.3f  Vanilla NN Error: %.3f   CNN Error: %.3f",
              err_base, err1, err_cnn))