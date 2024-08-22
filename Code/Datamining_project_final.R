# Data load
heart <- read.csv("heart.csv", header=TRUE) #heart.csv 경로
str(heart)


## 범주형 설명변수 분포
old.par <- par(mfrow=c(2,3))
for(i in c(2,3,6,7,9,11)){
  plot(as.factor(heart[[i]]), main=colnames(heart)[i])
}
par(old.par)


# 요약 통계량
summary(heart)


# NA값 처리 - 행 삭제
heart.new <- heart[!heart[[5]]==0,]
str(heart.new)
summary(heart.new)


# 산점도 행렬
pairs(~Age+RestingBP+Cholesterol+MaxHR+Oldpeak, data=heart.new)


# 연속형 설명변수 간 상관계수 확인
library(dplyr)
heart.cont <- select(heart.new, Age, RestingBP, Cholesterol, MaxHR, Oldpeak)
head(heart.cont)
cor(heart.cont)
library(corrplot)
corrplot(cor(heart.cont), method="number")


# 훈련 세트 / 테스트 세트
set.seed(1)
index <- sort(sample(1:nrow(heart.new), size=nrow(heart.new)*0.8))
train <- heart.new[index,]    
test <- heart.new[-index,]

####################### Support Vector Machine #######################
# Preprocessing for SVM Model
heart.new <- heart[!heart[[5]]==0,]

heart.new[heart.new$Sex=="F","Sex"]<-0
heart.new[heart.new$Sex=="M","Sex"]<-1
heart.new$Sex<-as.numeric(heart.new$Sex)
head(heart.new)

unique(heart.new$ChestPainType)
heart.new[heart.new$ChestPainType=="ATA","ChestPainType"]<--1
heart.new[heart.new$ChestPainType=="NAP","ChestPainType"]<-0
heart.new[heart.new$ChestPainType=="ASY","ChestPainType"]<-1
heart.new[heart.new$ChestPainType=="TA","ChestPainType"]<-2
heart.new$ChestPainType<-as.numeric(heart.new$ChestPainType)
unique(heart.new$ChestPainType)

unique(heart.new$ExerciseAngina)
heart.new[heart.new$ExerciseAngina=="Y","ExerciseAngina"]<-0
heart.new[heart.new$ExerciseAngina=="N","ExerciseAngina"]<-1
heart.new$ExerciseAngina<-as.numeric(heart.new$ExerciseAngina)


unique(heart.new$RestingECG)
heart.new[heart.new$RestingECG=="Normal","RestingECG"]<--1
heart.new[heart.new$RestingECG=="ST","RestingECG"]<-0
heart.new[heart.new$RestingECG=="LVH","RestingECG"]<-1
heart.new$RestingECG<-as.numeric(heart.new$RestingECG)

unique(heart.new$ST_Slope)
heart.new[heart.new$ST_Slope=="Down","ST_Slope"]<--1
heart.new[heart.new$ST_Slope=="Flat","ST_Slope"]<-0
heart.new[heart.new$ST_Slope=="Up","ST_Slope"]<-1
heart.new$ST_Slope<-as.numeric(heart.new$ST_Slope)

heart.new$HeartDisease<-as.factor(heart.new$HeartDisease)


# Split training / test data again
set.seed(1)
index<-sample(1:nrow(heart.new), size=nrow(heart.new)*0.8)
train<-heart.new[index,]
test<-heart.new[-index,]


# Fitting Kernel SVM to the training set
library(e1071)
result1 <- tune.svm(HeartDisease~.,data=train,cost=2^(0:4),kernel="linear")
result1$best.parameters
result2 <- tune.svm(HeartDisease~.,data=train, gamma=2^(-5:0),cost=2^(0:4),kernel="radial")
result2$best.parameters
classifier = svm(formula = HeartDisease ~ .,data = train,type = 'C-classification',kernel = 'linear', cost=1, cross=10)
classifier = svm(formula = HeartDisease ~ .,data = train,type = 'C-classification', gamma=0.125, kernel = 'radial', cost=1, cross=10)


# Predicting the test set results
svm.pred = predict(classifier, newdata = test[-12])


# Making the Confusion Matrix
cm = table(test[, 12], svm.pred)
cm
sum(cm[row(cm) == col(cm)])/sum(cm)


# AUC
library(e1071)
svm.result <- predict(classifier,train)
t <- table(svm.result,train$HeartDisease);t
sum(t[row(t) == col(t)])/sum(t)
library(Epi)
par(oma=c(1,1,1,1),mar=c(1,3,3,1))
ROC(test=svm.result,stat=train$HeartDisease,plot='ROC',AUC=T, main="Predict Heartdisease")


####################### Random Forest #######################
# Preprocessing for Random Forest
heart.new <- heart[!heart[[5]]==0,]

heart.new$HeartDisease = as.factor(heart.new$HeartDisease)
levels(heart.new$HeartDisease) = c("Normal", "heart disease")
str(train$HeartDisease)


# Split training / test data again
set.seed(1)
index <- sample(1:nrow(heart.new), size = nrow(heart.new)*0.8)     
train <- heart.new[index,]     
test <- heart.new[-index,]  


# Random Forest
library(randomForest)
(rf.heart<- randomForest(HeartDisease ~., data=train))

rf.heart$importance #변수중요도 / 변수가 정확도와 노드불순도 개선에 얼마만큼 기여하는지로 측정
varImpPlot(rf.heart, main="노드 불순도 개선량")

layout(matrix(c(1,2),nrow=1),width=c(4,1)) 
par(mar=c(5,4,4,0))
plot(rf.heart)
par(mar=c(5,0,4,2))
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(rf.heart$err.rate),col=1:4,cex=0.8,fill=1:4)


# ROC curve
library(ROCR)
rf.pred = predict(rf.heart,test[,-12],type = "class")
roc.pred =prediction(as.numeric(rf.pred),as.numeric(test[,12]))
plot(performance(roc.pred,"tpr","fpr"))
abline(a=0,b=1,lty=2,col="black")


# AUC
performance(roc.pred,"auc")@y.values


# Random Forest with 10-folds CV
library(cvTools)
library(foreach)
library(randomForest)

set.seed(1)
K = 10
R = 3
cv <- cvFolds(NROW(heart.new), K=K, R=R)

grid <- expand.grid(ntree=c(50, 100, 150, 200), mtry=c(1:11))
rf.result <- foreach(g=1:NROW(grid), .combine=rbind) %do% {
  foreach(r=1:R, .combine=rbind) %do% {
    foreach(k=1:K, .combine=rbind) %do% {
      validation_idx <- cv$subsets[which(cv$which == k), r]
      train <- heart.new[-validation_idx, ]
      validation <- heart.new[validation_idx, ]
      # 모델 훈련
      train$HeartDisease = as.factor(train$HeartDisease)
      m <- randomForest(HeartDisease ~.,
                        data=train,
                        ntree=grid[g, "ntree"],
                        mtry=grid[g, "mtry"])
      # 예측
      predicted <- predict(m, newdata=validation)
      # 성능 평가
      precision <- sum(predicted == validation$HeartDisease) / NROW(predicted)
      return(data.frame(g=g, precision=precision))
    }
  }
}


rf.result

library(plyr)
ddply(rf.result, .(g), summarize, mean_precision=mean(precision))

grid[8,]

heart.new <- heart[!heart[[5]]==0,]
heart.new$HeartDisease = as.factor(heart.new$HeartDisease)
set.seed(1)
index <- sample(1:nrow(heart.new), size = nrow(heart.new)*0.8)
train <- heart.new[index,]     
test <- heart.new[-index,]  

(rf.heart <- randomForest(HeartDisease ~., data=train,ntree=200,mtry=2, type="classification"))
layout(matrix(c(1,2),nrow=1),width=c(4,1)) 
par(mar=c(5,4,4,0))
plot(rf.heart)
par(mar=c(5,0,4,2))
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(rf.heart$err.rate),col=1:4,cex=0.8,fill=1:4)


# ROC curve
library(ROCR)
rf.pred = predict(rf.heart,test[,-12],type = "class")
roc.pred = prediction(as.numeric(rf.pred),as.numeric(test[,12]))
plot(performance(roc.pred,"tpr","fpr"))
abline(a=0,b=1,lty=2,col="black")


# AUC value
performance(roc.pred,"auc")@y.values

####################### Boosting #######################
# Preprocessing for Boosting
heart.new <- heart[!heart[[5]]==0,]

for (i in c(2,3,7,9,11)){
  heart.new[[i]] <- as.factor(heart.new[[i]])
}
str(heart.new)


# Split training / test data again
set.seed(1)
index <- sample(1:nrow(heart.new), size=nrow(heart.new)*0.8)
train <- heart.new[index,]
test <- heart.new[-index,]


# Boosting by gbm() with 10-folds CV - parameters : default
set.seed(1)
library(gbm)
boost.heart <- gbm(HeartDisease~., data=train, 
                    distribution="bernoulli", cv.folds=10)
boost.heart
summary(boost.heart, new=TRUE)


# prediction
heartdisease <- test$HeartDisease
boost.pred <- predict.gbm(boost.heart, test, ntrees=100, type="response")
boost.prediction <- rep(0, length(boost.pred))
boost.prediction[boost.pred>0.5] <- 1
table(heartdisease, boost.prediction)
accuracy <- (66+67)/(66+9+8+67) ; accuracy


# ROC Curve
library(ROCR)
pred <- prediction(boost.pred, heartdisease)
plot(performance(pred, "tpr", "fpr"))


# AUC
auc.boost <- performance(pred, measure="auc")
attributes(auc.boost)$y.values   # AUC = 0.9454


# grid search - hyperparemeter tuning
set.seed(1)
library(caret)
heart.new[[12]] <- as.factor(heart.new[[12]])
train.data <- train[,-12]
train.classes <- train$HeartDisease
traincontrol <- trainControl(method="cv", number=10, returnResamp="all", search="grid")
tunegrid <- expand.grid(n.trees=c(10,50,100,200,300),
                        interaction.depth=c(1,2,4,6,12),
                        shrinkage=c(0.01,0.1,0.2),
                        n.minobsinnode=c(5,10))
boost.grid <- caret::train(HeartDisease~., data=heart.new, method="gbm",
                    tuneGrid=tunegrid, trControl=traincontrol, verbose=FALSE)
plot(boost.grid)
boost.grid$bestTune


# Best Model
set.seed(1)
best.boost.heart <- gbm(HeartDisease~., data=train, 
                   distribution="bernoulli", cv.folds=10,
                   n.trees=50, interaction.depth=4, shrinkage=0.1)
best.boost.heart
summary(best.boost.heart)


# ROC Curve
best.boost.pred <- predict.gbm(best.boost.heart, test, ntrees=50, type="response")
best.pred <- prediction(best.boost.pred, heartdisease)
plot(performance(pred, "tpr", "fpr"))


# AUC
best.auc.boost <- performance(best.pred, measure="auc")
attributes(best.auc.boost)$y.values   # AUC = 0.9461
