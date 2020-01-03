library(rpart)


# read data
astronomy <- read.csv("~/Desktop/GI04/SY19/tp10/data/astronomy_train.csv",sep=",",header=TRUE)
head(astronomy)

# data splitting
class <- astronomy$class
astronomy <- subset(astronomy, select = -c(class))
View(astronomy)
View(class)
# data cleaning
## remove variable with constant value
s<-apply(astronomy, 2, sd)
ii<-which(s>0)
iix <- which(s<=0)
astronomy<-astronomy[, ii]
astronomy<-cbind(astronomy, class)
View(astronomy)

### data normalisation
astronomy<-scale(astronomy)
pca<-prcomp(astronomy)
lambda<-pca$sdev^2
pairs(pca$x[,1:5], col=y, pch=as.numeric(y))

plot(cumsum(lambda)/sum(lambda), xlab="q")

q<-100
X2<-scale(pca$x[,1:q])




# Arbre de décision
## Un arbre
n <- nrow(astronomy)
ntrain <- 2*n/3

idx.train = sample(n, ntrain)
ntrain <- length(idx.train)
ntest <- n - ntrain

astronomy$class <- as.factor(astronomy$class)
fit.tree <- rpart(class~., data = astronomy, 
                  subset = idx.train, method = "class",
                  control = rpart.control(xval = 10, minbucket = 10, cp = 0.00))
plot(fit.tree, margin = 0.0005)
text(fit.tree, minlength=0.5, cex=0.5, splits=TRUE)
library(rpart.plot)
rpart.plot(fit.tree, box.palette="RdBu", shadow.col="gray", fallen.leaves=FALSE)

tree.yhat <- predict(fit.tree, newdata = astronomy[-idx.train,], type = 'class')
tree.ytest <- astronomy[-idx.train, "class"]
matrix.conf.tree <- table(tree.ytest, tree.yhat)
matrix.conf.tree

err.tree <- 1 - sum(diag(matrix.conf.tree))/ntest
err.tree

## Tree pruned
printcp(fit.tree)
plotcp(fit.tree, minline = TRUE)

i.min <- which.min(fit.tree$cptable[,4])
cp.opt1 <- fit.tree$cptable[i.min, 1]

pruned_tree <- prune(fit.tree, cp=cp.opt1)
rpart.plot(pruned_tree, box.palette = "RdBu", shadow.col = "gray")

prunedTree.yhat <- predict(pruned_tree, newdata = astronomy[-idx.train,], type = 'class')
prunedTree.ytest <- astronomy[-idx.train, "class"]
matrix.conf.prunedTree <- table(prunedTree.ytest, prunedTree.yhat)
matrix.conf.prunedTree

err.tree <- 1 - sum(diag(matrix.conf.prunedTree))/ntest
err.tree

## plot of ROC curve of the pruned tree classifier
library(pROC)
prob <- predict(pruned_tree, newdata = astronomy[-idx.train,], type = 'prob')
roc_tree_pruned <- multiclass.roc(prunedTree.ytest, prob)

# Bagging
library(randomForest)
param <-ncol(astronomy) - 1
fit.bagged <- randomForest(class~., data=astronomy, subset=idx.train, mtry=param)
bagged.yhat <- predict(fit.bagged, newdata= astronomy[-idx.train,], type="response")
bagged.ytest <- astronomy[-idx.train, "class"]
CM.bagged <- table(bagged.yhat, bagged.ytest)
CM.bagged
err.bagged <- 1-mean(bagged.yhat == bagged.ytest)
err.bagged

# Random forest
fit.rf <- randomForest(class~., data=astronomy, subset=idx.train)
rf.ytest <- astronomy[-idx.train, "class"]
rf.yhat <- predict(fit.rf, newdata=astronomy[-idx.train,], type="response") 
CM.rf <- table(rf.ytest, rf.yhat)
err.rf <- 1-mean(rf.ytest== rf.yhat)
err.rf

# Logistic regression
fit.lr <- glm(class~., data=astronomy, subset=idx.train, family="binomial") 
summary(fit.lr)
pred.lr <- predict(fit.lr, newdata=astronomy[-idx.train,], type="response") 
lr.yhat <- pred.lr > 0.5
lr.ytest <- astronomy[-idx.train, "class"]
CM.lr <- table(lr.yhat, lr.ytest)
ntest <- length(lr.ytest)
err.lr <- 1 - sum(diag(CM.lr))/ntest
err.lr
# Gaussian mixture model (EM)

# SVM


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




