
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
