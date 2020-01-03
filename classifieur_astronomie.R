library(rpart)


# read data
astronomy <- read.csv("./data/astronomy_train.csv",sep=",",header=TRUE)
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






