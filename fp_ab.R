#################### PREPARE PROBLEM ####################
###############---------------------------###############

###############---------------------------###############
# A) Load Library
###############---------------------------###############
library(caTools)
library(rpart)
library(rpart.plot)
library(DataExplorer)
library(lattice)
library(gplots)
library(ggplot2)
library(ROCR)
library(nnet)
library(caret)

library(naivebayes)
library(dplyr)
library(data.table)
library(corrplot)
library(RColorBrewer)
library(nnet)
###############---------------------------###############
# B) Load Dataset
###############---------------------------###############

heart <- read.csv("F:/Kuliah/Tugas Semester 6/AB/Tugas/FP/heart.csv")

factorData <- copy(heart)
factorData$sex <- factor(factorData$sex)
factorData$cp <- factor(factorData$cp)
factorData$fbs <- factor(factorData$fbs)
factorData$restecg <- factor(factorData$restecg)
factorData$exang <- factor(factorData$exang)
factorData$ca <- factor(factorData$ca)
factorData$thal <- factor(factorData$thal)
factorData$target <- factor(factorData$target)

ExploreData <- copy(factorData)
ExploreData$thalach <- ifelse((ExploreData$thalach>0 & ExploreData$thalach<=130) , '<=130',ExploreData$thalach)
ExploreData$thalach <- ifelse((ExploreData$thalach>130) , '>130',ExploreData$thalach)
ExploreData$thalach<-as.factor(ExploreData$thalach)


###############---------------------------###############
# C) Function Box
###############---------------------------###############

# Function Evaluate Accuracy, Precision, Recall, F-Measure
evaluation <- function(model, data, atype) {
  hasilPrediciton <- predict(model, data, type=atype)
  table_evaluation <- table(hasilPrediciton, data$target)
  accuracy <- sum(diag(table_evaluation))/sum(table_evaluation)
  precision <- table_evaluation[1,1]/sum(table_evaluation[ ,1])
  recall <- table_evaluation[1,1]/sum(table_evaluation[1,])
  f <- 2* (precision*recall)/(precision+recall)
  
  cat(paste("Accuracy:\t", format(accuracy, digits=2), "\n",sep=" "))
  cat(paste("Precision:\t", format(precision, digits=2), "\n",sep=" "))
  cat(paste("Recall:\t\t", format(recall, digits=2), "\n",sep=" "))
  cat(paste("F-measure:\t", format(f, digits=2), "\n",sep=" "))
}

#################### EKSPLORASI DATA ####################
###############---------------------------###############
#Eksplorasi Age
ggplot(factorData) + aes(x=as.numeric(age), group=target, fill=target) + 
  geom_histogram(binwidth=1, color='black')
boxplot(factorData$age)
summary(factorData$age)
#Eksplorasi Sex
ggplot(data=factorData, aes(x=sex, fill=as.factor(target)))+
  geom_bar(alpha=.5)+
  ggtitle("sex disttribution")
#Eksplorasi cp
ggplot(factorData) + aes(x=as.numeric(cp), group=target, fill=target) + 
  geom_histogram(binwidth=1, color='black')

#Eksplorasi thalach
ggplot(data=ExploreData, aes(x=thalach, fill=as.factor(target)))+
  geom_bar(alpha=1)+
  ggtitle("thalach disttribution")

#Eksplorasi exang
ggplot(data=factorData, aes(x=exang, fill=as.factor(target)))+
  geom_bar(alpha=1)+
  ggtitle("exang disttribution")

corrplot(cor(heart), method= "circle")

# Menghitung missing value
sum(is.na(factorData))

#Menghapus Duplikasi
factorData <- unique(factorData)

####################  MEMBUAT Model  ####################
###############---------------------------###############
set.seed(10)
split <- sample.split(factorData$target, SplitRatio = 0.7)
train <- factorData[split,]
test <- factorData[!split,]

a<- data.frame("age"=40, "sex"=1, "cp"=3, "trestbps"=118, "chol"= 200, 
               "fbs"= 0,"restecg"= 0, "thalach"= 100, "exang"= 1, 
               "oldpeak"= 2.2, "slope"= 2, "ca"= 3, "thal"= 1)
a$sex <- factor(a$sex)
a$cp <- factor(a$cp)
a$fbs <- factor(a$fbs)
a$restecg <- factor(a$restecg)
a$exang <- factor(a$exang)
a$ca <- factor(a$ca)
a$thal <- factor(a$thal)
colSums(is.na(heart))

#bayes
bayes_model = naive_bayes(target~., data=train)#model

bayes_predict = predict(bayes_model, newdata=train, type = "class")#prediksi
table(train$target, bayes_predict)
evaluation(bayes_model, train, "class")

#Neural Network
nnmodel <- nnet(target ~ ., data = train, size = 10, maxit = 500)
nnpredict <- predict(nnmodel, newdata = train, type = 'class')
evaluation(nnmodel,train, "class")
table(test$target, nnpredict1)

#test dengan data testing
bayes_predict1 = predict(bayes_model, newdata=test, type = "class")
table(test$target, bayes_predict1)
evaluation(bayes_model,test, "class")

bayes_predict2 = predict(bayes_model, newdata=a, type = "class")#prediksi
bayes_predict2

nnpredict1 = predict(nnmodel, newdata = test)
evaluation(nnmodel,test, "class")
####################  MEMBUAT ROCR   ####################
###############---------------------------###############

prbayes <- prediction(as.numeric(bayes_predict), as.numeric(train$target))
prfbayes <- performance(prbayes, measure = "tpr", x.measure = "fpr")
ddbayes <- data.frame(FP = prfbayes@x.values[[1]], TP = prfbayes@y.values[[1]])

prbayes1 <- prediction(as.numeric(bayes_predict1), as.numeric(test$target))
prfbayes1 <- performance(prbayes1, measure = "tpr", x.measure = "fpr")
ddbayes1 <- data.frame(FP = prfbayes1@x.values[[1]], TP = prfbayes1@y.values[[1]])

prnn <- prediction(as.numeric(nnpredict), as.numeric(train$target))
prfnn <- performance(prnn, measure = "tpr", x.measure = "fpr")
ddnn <- data.frame(FP = prfnn@x.values[[1]], TP = prfnn@y.values[[1]])

prnn1 <- prediction(as.numeric(nnpredict1), as.numeric(test$target))
prfnn1 <- performance(prnn1, measure = "tpr", x.measure = "fpr")
ddnn1 <- data.frame(FP = prfnn1@x.values[[1]], TP = prfnn1@y.values[[1]])

#plot

g <- ggplot() + 
  geom_line(data = ddbayes, aes(x = FP, y = TP, color = 'Bayes Predict Train')) + 
  geom_line(data = ddbayes1, aes(x = FP, y = TP, color = 'Bayes Predict Test')) +
  geom_line(data = ddnn, aes(x = FP, y = TP, color = 'NN Predict Train')) +
  geom_line(data = ddnn1, aes(x = FP, y = TP, color = 'NN Predict Test')) +
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1)) +
  ggtitle('ROC Curve') + 
  labs(x = 'False Positive Rate', y = 'True Positive Rate') 


g+scale_colour_manual(name = 'Classifier',values = 
                        c('Bayes Predict Train'='#56B4E9',
                        'Bayes Predict Test'='#009E73' ,
                        'NN Predict Train'='#0000FF',
                        'NN Predict Test'='#FF0000'))
g



