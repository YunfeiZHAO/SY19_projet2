
classifieur_astronomie <- function(dataset) {
  library(nnet)
  load("env.Rdata")

  class <- dataset$class
  # remove constant
  X <- subset(dataset, select = -c(class, objid, rerun))
  #scaling
  X <- scale(X)
  # PCA
  astronomy <- data.frame(X %*% pca.rotation)
  # astronomy<-cbind(astronomy, class)
  # View(astronomy)
  # astronomy$class <- as.factor(astronomy$class)
  predictions <- predict(fit.MLR.final, newdata=astronomy) 
  return(predictions)
}

regresseur_mais <- function(dataset) {
  library("kernlab")
  load("env.Rdata")

  dataset$X = NULL
  dataset$IRR = as.factor(dataset$IRR)
  dataset = dataset[, -reg_corr_cols]
  if (ncol(dataset) != 44)
    print("ERREUR dans le nombre de colonnes. Le dataset n'est pas dans le même format que celui fourni.")

  predictions = predict(best_reg_model, newdata=dataset)

  return(predictions)
}


classifieur_images <- function(list) {
  load("env.RData")
  model <- unserialize_model(model.best, custom_objects = NULL, compile = TRUE)
  len<-length(list)
  dim_images <- c(50, 50)
  r <- rep(0, length(list))
  for(i in 1:len){
    images <- array(0, dim=c(2, dim_images[1], dim_images[2], 3))
    images[1,,,] <- image_to_array(image_load(path=list[i], target_size=dim_images, interpolation = "bilinear"))
    p <- predict(model, images[,,,], batch_size = 2)[1,]
    r[i] <- which.max(p)
  }
  r[r==1] <- 'car'
  r[r==2] <- 'cat'
  r[r==3] <- 'flower'
  return(as.factor(r))
}
#i <- c("/Users/yunfei/Desktop/GI04/SY19/tp10/SY19_projet2/data/images_train/car/car_train_2.jpg","/Users/yunfei/Desktop/GI04/SY19/tp10/SY19_projet2/data/images_train/cat/cat_train_8.jpg" )
#A = classifieur_images(i)
#save(fit.MLR.final, pca.rotation, best_reg_model, model.best, reg_corr_cols, file = "env.RData")
