install_keras()

library(keras)
library(formattable)
library(tidyverse)
library(imager)

getwd()
setwd("/Users/yunfei/Desktop/GI04/SY19/tp10/SY19_projet2/")


##############################1 NN (pas testé)###############################
# read data
read.images <- function(path = "./") {
  fns <- list.files(path)
  res <- NULL
  for(i in fns) {
    fn <- paste(path, i, sep = "")
    im <- load.image(fn)
    ## change to gray image of size 100*100
    im <- resize(im, size_x = 100L, size_y = 100L,
                 size_z = 1L, size_c = 1L)
    im <- as.array(im)
    im <- matrix(im, 1, product(dim(im)))
    ifelse(!is.null(res),
           res <- rbind(res, im), res <- im)
  }
  return(res)
}

car <- read.images("./data/images_train/car/")
View(car)
cat <- read.images("./data/images_train/cat/")
View(cat)
flower <- read.images("./data/images_train/flower/")
View(flower)

# data splitting
n <- nrow(car) + nrow(cat) + nrow(flower)
y1 <- rep(1, nrow(car))
y2 <- rep(2, nrow(cat))
y3 <- rep(3, nrow(flower))
y <- c(y1, y2, y3)

ntrainImg <- 2*n/3
idx.trainImg = sample(n, ntrain)
ntrainImg <- length(idx.trainImg)
ntestImg <- n - ntrainImg

ytrainImg <- y[idx.trainImg]
ytestImg <- y[-idx.trainImg]

data.image<- rbind(car, cat, flower)

xtrainImg <- data.image[idx.trainImg,]
xtestImg <- data.image[-idx.trainImg,]


#kera
model <- keras_model_sequential()
model %>% layer_dense(units = 30, activation = 'relu', input_shape = 10000) %>% 
  layer_dense(units = 20, activation = 'relu',name="cache1") %>% 
  layer_dense(units = 5, activation = 'relu',name="cache2") %>% 
  layer_dense(units = 3, activation = 'linear',name="sortie")
model %>% compile(loss = 'mean_squared_error', optimizer = optimizer_rmsprop())

history <- model %>% fit(xtrainImg, ytrainImg, epochs = 2000, batch_size = 30)
pred <- predict(model, xtestImg)



test<-car[1,]
test<-t(matrix(test, 100, 100))
test<-matrix(test, 100, 100, byrow=TRUE)
test<-matrix(test, 100, 100)
image(test)




######################tool functions####################
product <- function(vec){
  out <- 1
  for(i in 1:length(vec)){
    out <- out*vec[i]
  }
  out
}

#  a implementer
reverse <- function(M){
  out <- 1
  for(i in 1:length(vec)){
    out <- out*vec[i]
  }
  out
}


#################################### Neural network ####################################
############################# 1, data loading ###############################
# list of objects to modle
object_list <- c("car", "cat", "flower")

# image size to scale down to (original images are 100 x 100 px)
img_width <- 100
img_height <- 100
target_size <- c(img_width, img_height)


# RGB = 3 channels
channels <- 3

# path to image folders
image_files_path <- "./data/images_train/"


data_gen = image_data_generator(
  rescale = 1/255, #,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  #horizontal_flip = TRUE,
  #fimaill_mode = "nearest"
  validation_split = 0.3
)

# training images
train_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = object_list,
                                                    seed = 42,
                                                    subset = "training")

# validation images
validation_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = object_list,
                                                    seed = 42,
                                                    subset = "validation")
table(factor(train_image_array_gen$classes))
table(factor(validation_image_array_gen$classes))
# number of training samples
ntrain <- train_image_array_gen$n
nvalidation <- validation_image_array_gen$n


#----------------------------
# NN Model
#----------------------------
model2 <- keras_model_sequential() 
model2 %>% 
  layer_dense(units = 50, activation = 'relu', input_shape = c(100,100,3), kernel_regularizer = regularizer_l2(l=0.1)) %>% 
  layer_dropout(rate = 0.4)%>%
  layer_dense(units = 30, activation = 'relu', kernel_regularizer = regularizer_l2(l=0.1)) %>% 
  layer_dropout(rate = 0.4)%>%
  layer_dense(units = 20, activation = 'relu', kernel_regularizer = regularizer_l2(l=0.1)) %>% 
  layer_dropout(rate = 0.4)%>%
  layer_dense(units = 3, activation = 'softmax')
#Compiling the Model
model2 %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)
#Summary of the Model and its Architecture
summary(model2)

#----------------------------
# Fitting
#----------------------------

hist2 <- model2 %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  epochs = 50, 
  steps_per_epoch = as.integer(ntrain / train_image_array_gen$batch_size), 
  
  
  # validation data
  validation_data = validation_image_array_gen,
  validation_steps = as.integer(nvalidation / validation_image_array_gen$batch_size),
  
  # print progress
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint("./checkpoints/cnn_checkpoints.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = "./checkpoints/logs")
  )
)

plot(hist)
#------------------predictions---------------------------
predictions1 <- hist$model %>% predict_generator(validation_image_array_gen, steps=validation_image_array_gen$n/batch_size+1, verbose=1)

colnames(predictions1) <- c("car","cat","flower")

pred_labels<-colnames(predictions1)[apply(predictions1,1,which.max)]
proba<-apply(predictions1, 1, max)
stat_df <- as.data.frame(cbind(validation_image_array_gen$filenames, round(proba*100,2), pred_labels))
colnames(stat_df) <- c("filename","proba","class")
stat_df

validation_labels <- c("car","cat","flower")[validation_image_array_gen$classes + 1]
pred_labels

matrix.conf.CNN <- table(pred_labels, validation_labels)
matrix.conf.CNN

ntest <- length(pred_labels)
err.CNN <- 1 - sum(diag(matrix.conf.CNN))/ntest
err.CNN
#----------------------------
# CNN Model
#----------------------------
model1<-keras_model_sequential()
#Configuring the Model
model1 %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3),padding="same",
                input_shape=c(100,100,3)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter=48 , kernel_size=c(3,3),padding="same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3) ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  #flatten the input
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  #output layer-3 classes-3 units
  layer_dense(3) %>%
  
  #applying softmax nonlinear activation function to the output layer to calculate
  #cross-entropy
  layer_activation("softmax") #for computing Probabilities of classes-"logit(log probabilities)


#Optimizer -rmsProp to do parameter updates 
opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)


#Compiling the Model
model1 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

#Summary of the Model and its Architecture
summary(model1)

#----------------------------
# Fitting
#----------------------------
hist1 <- model1 %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  batch_size = 30,
  steps_per_epoch = as.integer(ntrain / train_image_array_gen$batch_size), 
  epochs = 100, 
  
  # validation data
  validation_data = validation_image_array_gen,
  validation_steps = as.integer(nvalidation / validation_image_array_gen$batch_size),
  
  # print progress
  verbose = 2,
  callbacks = list(
    # save best model after every epoch
    callback_model_checkpoint("./checkpoints/cnn_checkpoints.h5", save_best_only = TRUE),
    # only needed for visualising with TensorBoard
    callback_tensorboard(log_dir = "./checkpoints/logs")
  )
)

plot(hist)
#------------------predictions---------------------------
predictions1 <- hist$model %>% predict_generator(validation_image_array_gen, steps=validation_image_array_gen$n/batch_size+1, verbose=1)

colnames(predictions1) <- c("car","cat","flower")

pred_labels<-colnames(predictions1)[apply(predictions1,1,which.max)]
proba<-apply(predictions1, 1, max)
stat_df <- as.data.frame(cbind(validation_image_array_gen$filenames, round(proba*100,2), pred_labels))
colnames(stat_df) <- c("filename","proba","class")
stat_df

validation_labels <- c("car","cat","flower")[validation_image_array_gen$classes + 1]
pred_labels

matrix.conf.CNN <- table(pred_labels, validation_labels)
matrix.conf.CNN

ntest <- length(pred_labels)
err.CNN <- 1 - sum(diag(matrix.conf.CNN))/ntest
err.CNN


#------------------save/load model------------------------
save_model_hdf5(model,filepath = "/Users/yunfei/Desktop/GI04/SY19/tp10/SY19_projet2/model_cnn", overwrite = TRUE,
                include_optimizer = TRUE)

model <- load_model_hdf5(filepath = "C:/Users/xingjian/Desktop/sy19_projet/model_cnn",compile = T)