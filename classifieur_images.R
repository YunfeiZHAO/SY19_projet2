
##############################3 Images naturelles###############################
library(keras)
library(imager)

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

car <- read.images("~/Desktop/GI04/SY19/tp10/data/images_train/car/")
View(car)
cat <- read.images("~/Desktop/GI04/SY19/tp10/data/images_train/cat/")
View(cat)
flower <- read.images("~/Desktop/GI04/SY19/tp10/data/images_train/flower/")
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