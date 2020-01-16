
classifieur_astronomie <- function(dataset) {
  library(nnet)
  # Chargement de l’environnement
  load("env.Rdata")
  # Mon algorithme qui renvoie les prédictions sur le jeu de données
  # ‘dataset‘ fourni en argument. # ...
  class <- dataset$class
  # remove constant
  X <- subset(dataset, select = -c(class, objid, rerun))
  #scaling
  X <- scale(X)
  # PCA
  astronomy<-data.frame(X%*%pca.rotation)
  #astronomy<-cbind(astronomy, class)
  View(astronomy)
  #astronomy$class <- as.factor(astronomy$class)
  predictions<-predict(fit.MLR.norm, newdata=astronomy) 
  return(predictions)
}


'''
setwd("/Users/yunfei/Desktop/GI04/SY19/tp10/SY19_projet2/")
data <- read.csv("./data/astronomy_train.csv",sep=",",header=TRUE)
n <- nrow(data)
ntest <- 2000
idx <- sample(n, ntest)
prediction <- classifieur_astronomie(data[idx,])
matrix.conf <- table(prediction, data[idx, "class"])
matrix.conf
err <- 1 - sum(diag(matrix.conf))/ntest
err
'''
