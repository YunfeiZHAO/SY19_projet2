library(kknn)
library(rpart)
library(pROC)
library(MASS)
library(corrplot)
library(nnet)

##CARET choose variable
getwd()
setwd("/Users/yunfei/Desktop/GI04/SY19/tp10/SY19_projet2/")

# read data
astronomy <- read.csv("./data/astronomy_train.csv",sep=",",header=TRUE)
head(astronomy)

# data splitting
n <- nrow(astronomy)
ntrain <- 2*n/3
idx.train = sample(n, ntrain)
ntrain <- length(idx.train)
ntest <- n - ntrain

# Nested cross validation
n<-nrow(astronomy)
n_folds<-12

# data cleaning
## remove variable with constant value
class <- astronomy$class
astronomy <- subset(astronomy, select = -c(class))
s<-apply(astronomy, 2, sd)
ii<-which(s>0)
iix <- which(s<=0)
astronomy<-astronomy[, ii]
astronomy<-cbind(astronomy, class)
astronomy$class <- as.factor(astronomy$class)
# visualisation of class
#View(astronomy)
hist(as.numeric(astronomy$class)) 

#Correlation check
correlations <- cor(subset(astronomy, select = -c(class)),method="pearson")
corrplot(correlations, number.cex = .9, method = "circle", type = "full", tl.cex=1,tl.col = "black")

## data normalisation
scaled.X <- scale(subset(astronomy, select = -c(class)))
pca<-prcomp(scaled.X)
#plot(cumsum(lambda)/sum(lambda), xlab="q")
# take fisrt 10 variable
pca.X <- data.frame(pca$x[,1:12])

astronomy.normalised<-cbind(pca.X, class)
astronomy.normalised$class <- as.factor(class)
correlations.pca <- cor(subset(astronomy.normalised, select = -c(class)),method="pearson")
corrplot(correlations.pca, number.cex = .9, method = "circle", type = "full", tl.cex=1,tl.col = "black")

#----------------------------
# KNN
#----------------------------
fit.KNN <- kknn(class~., astronomy[idx.train,], astronomy[-idx.train,], distance = 1, kernel = "triangular")
pred.KNN <- fit.KNN$fitted.values
matrix.conf.KNN<-table(pred.KNN, astronomy[-idx.train, "class"])
matrix.conf.KNN
err.KNN <- 1 - sum(diag(matrix.conf.KNN))/ntest
err.KNN
#0.1019796
#----------------------------
# LDA
#----------------------------
##Model building##
fit.LDA<- lda(class~.,data=astronomy.normalised[idx.train,])
pred.LDA<-predict(fit.LDA, newdata=astronomy.normalised[-idx.train,])
matrix.conf.LDA<-table(pred.LDA$class, astronomy[-idx.train, "class"])
matrix.conf.LDA
err.LDA <- 1 - sum(diag(matrix.conf.LDA))/ntest
err.LDA
#0.1313737

roc_LDA <- multiclass.roc(astronomy.normalised[-idx.train,]$class, pred.LDA$posterior, plot=TRUE)
# 0.975

#----------------------------
# QDA
#----------------------------
fit.QDA<- qda(class~.,data=astronomy.normalised[idx.train,]) 
pred.QDA<-predict(fit.QDA, newdata=astronomy.normalised[-idx.train,]) 
matrix.conf.QDA <-table(pred.QDA$class, astronomy[-idx.train, "class"]) 
matrix.conf.QDA
1-sum(diag(matrix.conf.QDA))/ntest
# 0.02639472

fit.QDA<- qda(class~dec + u + g + run + camcol + field + redshift + fiberid, data=astronomy[idx.train,]) 
pred.QDA<-predict(fit.QDA, newdata=astronomy[-idx.train,]) 
matrix.conf.QDA <-table(pred.QDA$class, astronomy[-idx.train, "class"]) 
matrix.conf.QDA
1-sum(diag(matrix.conf.QDA))/ntest
#----------------------------
# naive Bayes
#----------------------------
library(naivebayes) 
fit.naive<- naive_bayes(class~.,astronomy.normalised[idx.train,]) 
pred.naive<-predict(fit.naive, newdata=astronomy.normalised[-idx.train,]) 
pred.naive <- as.factor(pred.naive)
matrix.conf.naive<-table(pred.naive, astronomy.normalised[-idx.train, "class"])
matrix.conf.naive
1-sum(diag(matrix.conf.naive))/ntest
# 0.1445711


#----------------------------
# Multinomial Logistic Regression
#----------------------------
###############final################
fit.MLR.final <- multinom(class ~ ., data = astronomy.normalised)
summary(fit.MLR.final)
pred.MLR.final <- predict(fit.MLR.final, newdata=astronomy.normalised) 
matrix.conf.MLR.final <- table(pred.MLR.final, astronomy.normalised[, "class"])
matrix.conf.MLR.final
err.MLR.final <- 1 - sum(diag(matrix.conf.MLR.final))/n
err.MLR.final
###############final################

fit.MLR.norm <- multinom(class ~ ., data = astronomy.normalised[idx.train,])
summary(fit.MLR.norm)
pred.MLR.norm <- predict(fit.MLR.norm, newdata=astronomy.normalised[-idx.train,]) 
matrix.conf.MLR.norm <- table(pred.MLR.norm, astronomy.normalised[-idx.train, "class"])
matrix.conf.MLR.norm
err.MLR.norm <- 1 - sum(diag(matrix.conf.MLR.norm))/ntest
err.MLR.norm
# 0.00539892
#0.0119976

# final  value 146.843437 
# stopped after 100 iterations
pred.MLR.norm.self <- predict(fit.MLR.norm, newdata=astronomy.normalised[idx.train,]) 
matrix.conf.MLR.norm.self <- table(pred.MLR.norm.self, astronomy.normalised[idx.train, "class"])
matrix.conf.MLR.norm.self
err.MLR.norm.self <- 1 - sum(diag(matrix.conf.MLR.norm.self))/ntrain
err.MLR.norm.self
# 0.0060006

fit.MLR <- multinom(class ~ ra + u + r + i + z + run + redshift + mjd, data = astronomy[idx.train,])
summary(fit.MLR)
pred.MLR <- predict(fit.MLR, newdata=astronomy[-idx.train,]) 
matrix.conf.MLR <- table(pred.MLR, astronomy[-idx.train, "class"])
matrix.conf.MLR
err.MLR <- 1 - sum(diag(matrix.conf.MLR))/ntest
err.MLR
#0.01439712
#----------------------------
# Arbre de décision
#----------------------------
## Un arbre
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
# 0.01019796

#----------------------------
# Tree pruned
#----------------------------
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

err.prunedTree <- 1 - sum(diag(matrix.conf.prunedTree))/ntest
err.prunedTree
# 0.01019796
## plot of ROC curve of the pruned tree classifier
prob <- predict(pruned_tree, newdata = astronomy[-idx.train,], type = 'prob')
roc_tree_pruned <- multiclass.roc(prunedTree.ytest, prob)
roc_tree_pruned
#----------------------------
# Bagging
#----------------------------
library(randomForest)
param <-ncol(astronomy) - 1
fit.bagged <- randomForest(class~., data=astronomy, subset=idx.train, mtry=param)
bagged.yhat <- predict(fit.bagged, newdata= astronomy[-idx.train,], type="response")
bagged.ytest <- astronomy[-idx.train, "class"]
CM.bagged <- table(bagged.yhat, bagged.ytest)
CM.bagged
err.bagged <- 1-mean(bagged.yhat == bagged.ytest)
err.bagged
# 0.01079784
#----------------------------
# Random forest
#----------------------------
fit.rf <- randomForest(class~., data=astronomy, subset=idx.train)
rf.ytest <- astronomy[-idx.train, "class"]
rf.yhat <- predict(fit.rf, newdata=astronomy[-idx.train,], type="response") 
CM.rf <- table(rf.ytest, rf.yhat)
err.rf <- 1-mean(rf.ytest== rf.yhat)
err.rf
# 0.01079784
#----------------------------
# SVM
#----------------------------
## data normalisation
class <- astronomy$class
SVM.X <- subset(astronomy, select = -c(class))
SVM.X <- scale(SVM.X)

SVM.X.train <- SVM.X[idx.train,]
SVM.X.test <- SVM.X[-idx.train,]

SVM.Y.train <- class[idx.train]
SVM.Y.test <- class[-idx.train]

#----------------------------
#  SVM linéaire
#----------------------------
library("kernlab")

### SVM avec noyau linéaire
# Réglage de C par validation croisée
CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
N<-length(CC)
M<-10 # nombre de répétitions de la validation croisée
err<-matrix(0,N,M)
for(k in 1:M){
  for(i in 1:N){
    err[i,k]<-cross(ksvm(x=SVM.X.train,y=SVM.Y.train,type="C-svc",kernel="vanilladot",C=CC[i],cross=5))
  }
}
Err<-rowMeans(err)
plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")

# Calcul de l'erreur de test avec la meilleure valeur de C
minCC <- CC[which.min(Err)]
svmfit<-ksvm(x=SVM.X.train,y=SVM.Y.train,type="C-svc",kernel="vanilladot",C=minCC)
SVM.pred<-predict(svmfit,newdata=SVM.X.test)
matrix.conf.svm<-table(SVM.Y.test,SVM.pred)
err.svm<-1-sum(diag(matrix.conf.svm))/ntest
err.svm
#0.01019796
#----------------------------
#  SVM non linéaire
#----------------------------

################# SVM avec noyau gaussien ####################
# Calcul de l'erreur de test avec la meilleure valeur de C
svmfit<-ksvm(x=SVM.X.train,y=SVM.Y.train,type="C-svc",kernel="rbfdot",C=minCC)
SVM.pred.rbfdot<-predict(svmfit,newdata=SVM.X.test)
matrix.conf.svm.rbfdot<-table(SVM.pred.rbfdot, SVM.Y.test)
matrix.conf.svm.rbfdot
err.svm.rbfdot<-1-sum(diag(matrix.conf.svm.rbfdot))/ntest
err.svm.rbfdot
#0.04259148
############# SVM avec noyau polynomial ############
svmfit<-ksvm(x=SVM.X.train,y=SVM.Y.train,type="C-svc",kernel="polydot",C=minCC)
SVM.pred.polydot<-predict(svmfit,newdata=SVM.X.test)
matrix.conf.svm.polydot<-table(SVM.pred.polydot, SVM.Y.test)
err.svm.polydot<-1-sum(diag(matrix.conf.svm.polydot))/ntest
err.svm.polydot


########################### save Rdata #################################
save.image("./classifier_astronomie.RData")


load("classifier_astronomie.RData")

pca.rotation <- pca$rotation
save(fit.MLR.final, pca.rotation, file = "env.RData")

