library(tree)
library(randomForest)
library(gbm)
library(e1071)
library(caret)
library(class)

data<-read.csv("classif1.txt")
name<-c("x1","x2","x3","x4","fraud")
colnames(data)<-name

# Creating train and test sets

set.seed(1)
train_num<-sample(1:length(data$x1),0.7*length(data$x1))
train<-data[train_num,]
test<-data[-train_num,]


# Trying classification method
# 
train$fraud<-as.factor(train$fraud)
# 
# test$fraud<-as.numeric(test$fraud)
test$fraud<-as.factor(test$fraud)
# 
# 
# train$fraud<-as.numeric(train$fraud)
# train$fraud[train$fraud!=1]<-0
# train$fraud[is.na(train$fraud)]<-0
# train$fraud<-as.factor(train$fraud)
# test$fraud<-as.numeric(test$fraud)
# test$fraud[test$fraud!=1]<-0
# test$fraud[is.na(test$fraud)]<-0
# test$fraud<-as.factor(test$fraud)

#train$dear<-NULL
#test$dear<-NULL


tree.train<-tree(fraud~.,train)
summary(tree.train)

plot(tree.train)
text(tree.train,pretty=0)

tree.test<-predict(tree.train,test,type = "class")
table(tree.test,test$fraud)

(2+7)/(224+7+2+179)

# Now let's try some pruning methods

cv.names_tree<-cv.tree(tree.train,FUN=prune.misclass)
names(cv.names_tree)

cv.names_tree$size
cv.names_tree$dev
cv.names_tree$k
cv.names_tree$method

par(mfrow=c(1,2))
plot(cv.names_tree$size,cv.names_tree$dev,type="b")
plot(cv.names_tree$k,cv.names_tree$dev,type="b")

prune.names_tree<-prune.misclass(tree.train,best=14)
plot(prune.names_tree)
text(prune.names_tree,pretty=0)

tree.test.pred<-predict(prune.names_tree,test,type = "class")
table(tree.test.pred,test$fraud)

#No difference because best was at size 14

# Attempting a bagging model

bag.email<-randomForest(fraud~.,data=train,mtry=length(colnames(data)),importance=TRUE)
bag.email

bag.yhat<-predict(bag.email,newdata=test)
plot(bag.yhat,test$fraud)
abline(0,1)
bag.err<-mean(bag.yhat!=test$fraud)
bag.err

library(ipred)
bag.email<-bagging(fraud~.,data=train,coob=TRUE,nbagg=1000)
bag.yhat<-predict(bag.email,newdata = test)
bag.err<-mean(bag.yhat!=test$fraud)
bag.err

table(bag.yhat,test$fraud)

# Attempting a random forest model

pred<-ncol(train)-1

rf.email<-randomForest(fraud~.,data=train,mtry=round(sqrt(pred)),importance=TRUE)
rf.email

rf.trainhat<-predict(rf.email,newdata=train[,-5])
table(rf.trainhat,train$fraud)

rf.yhat<-predict(rf.email,newdata=test[,-5])
plot(rf.yhat,test$fraud)
abline(0,1)
rf.err<-mean(rf.yhat!=test$fraud)
table(rf.yhat,test$fraud)
rf.err

varImpPlot(rf.email)

# Excellent result of about .09%

# Attempting a boosting model

train$fraud<-as.numeric(as.character(train$fraud))
#train$fraud<-as.integer(train$fraud)

set.seed(1)
boost.email<-gbm(fraud~.,data=train,distribution = "bernoulli",n.trees=50000,interaction.depth = 4)
boost.email
summary(boost.email)

par(mfrow=c(1,2))
plot(boost.email,i="rm")
plot(boost.email,i="lstat")

boost.yhat=predict(boost.email,newdata=test,n.trees=50000)
boost.yhat<-plogis(2*boost.yhat)
boost.err<-mean(boost.yhat!=test$fraud)

shrink.boost.email<-gbm(fraud~.,data=train,distribution = "bernoulli",n.trees=50000,interaction.depth = 4,
                        shrinkage = 0.1,verbose = F)
shrink.boost.yhat=predict(shrink.boost.email,newdata=test,n.trees=50000)
shrink.boost.err<-mean(shrink.boost.yhat!=test$fraud)

boost.err
shrink.boost.err

train$fraud<-as.factor(train$fraud)
# Attempting a SVM

train.x<-as.matrix(train[,-5])
train.y<-as.matrix(as.numeric(as.character(train[,5])))
test.x<-as.matrix(test[,-5])
test.y<-as.matrix(as.numeric(as.character(test[,5])))

par(mfrow=c(1,1))
plot(train.x[,c(3,4)],col=as.factor(train.y))
legend(5,-7,legend = c("Class 0","Class 1"),col=c("black","red"),pch = 1:1)

svmfit<-svm(fraud~.,data=train,kernel="linear",cost=10,scale=FALSE)
plot(svmfit,train)

set.seed(1)
tune.out<-tune(svm,fraud~.,data=train,kernel="linear",ranges=list(cost=c(1,5,10,100,200,300,350,400,500)))
summary(tune.out)

bestmod<-tune.out$best.model
summary(bestmod)

ypred<-predict(bestmod, subset(test,select = -fraud))
table(predict=ypred,truth=test$fraud)

# Trying a non linear SVM

svmfit<-svm(fraud~.,data=train,kernel="radial",gamma=1,cost=1)
plot(svmfit,train)

set.seed(1)
#tune.out<-tune(svm,fraud~.,data=train,kernel="radial",ranges= list(cost=c(0.1,1,2,5,10,100,300,500,1000)),gamma=c(0.5,1,2,3,4))

tune.out=tune(svm , fraud~., data=train, kernel ="radial",
              ranges =list(cost=c(0.1 ,1 ,10 ,100 ,1000),
                           gamma=c(0.5,1,2,3,4) ))
summary(tune.out)


table(truth=test$fraud,pred=predict(tune.out$best.model,newdata = subset(test,select = -fraud)))

# Attempting a KNN

n<- length(data$fraud)
k.vec <- c(1,2,3,4,5,6,7,8,9,10,11,13)
nk <- length(k.vec)
p<-ncol(data)-1

cv <- numeric(nk)

for(j in 1:nk)
{
  k <- k.vec[j]
  for(i in 1:n)
  { 
    x.out <- data[i,-(p)]
    y.out <- data[i, (p)]
    
    yhat.out <- as.numeric(as.character(knn(data[-i,-(p)], x.out, data[-i,p], k=k)))
    cv[j] <- cv[j] + mean((y.out-yhat.out)^2)
  }
  
  cv[j] <- cv[j]/n 
}

plot(k.vec, cv, type='b', xlab='k', ylab='cv(k)',main = "CV for value of k")

train.x<-as.matrix(train[,-5])
train.y<-as.matrix(as.numeric(as.character(train[,5])))
test.x<-as.matrix(test[,-5])
test.y<-as.matrix(as.numeric(as.character(test[,5])))

knn<-knnreg(train.x,train.y,k=1)

knn.pred<-predict(knn,newdata = train.x)
knn.pred<-knn(train.x,train.x,train.y,k=1)
knn_mse_train<-mean((knn.pred!=train.y))

knn.pred<-predict(knn,newdata = test.x)
knn.pred<-knn(train.x,test.x,train.y,k=1)
knn_mse_test<-mean((knn.pred!=test.y))
table(knn.pred,test.y)

paste("The train mse for knn is ",knn_mse_train)
paste("The test mse for knn is ",knn_mse_test)




