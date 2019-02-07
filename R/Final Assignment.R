library(dbplyr)
library(caTools)
library(tidyr)
library(tidyverse)
library(caret)
library(kernlab)
library(rpart)
library(randomForest)
#library(ggplot2)
library(lattice)
library(knitr)
#library(rmarkdown)
library(shiny)
library(class)
library(rattle)
library(klaR)
library(e1071)


#setwd("C:\Users\magic\Desktop\R Code")

#Read and View Data
read_data<-function(){
  data<-read.csv('dataset.csv')
  #Remove CustomerID column
  data<-data[,-1]
  assign('data',data,envir = globalenv())
}

##View Structure of the data
view_structure<-function(){
  print("\n")
  str(data)
  summary(data)
}

#Check For Null values
check_na<-function(){
  #print('Checking for Null values.....')
  nulls<-sapply(data,function(x)sum(is.na(x)))
  print(nulls)
  assign('nulls',nulls,envir = globalenv())
  
}

#Handle Tenure's Null Values
tenure_handle<-function(){
  tenure_mean<-mean(data$tenure,na.rm = TRUE)
  assign('tenure_mean',tenure_mean,envir = globalenv())
  data$tenure[is.na(data$tenure)]<-tenure_mean
  assign('data',data,envir = globalenv())
  
}

#Encode and scale Dataset cols
pre_preocessing<-function(){
  #Label Encoding Factor Columns
  data$gender<-as.factor(data$gender)
  data$Partner<-as.factor(data$Partner)
  data$Dependents<-as.factor(data$Dependents)
  data$PhoneService<-as.factor(data$PhoneService)
  data$MultipleLines<-as.factor(data$MultipleLines)
  data$InternetService<-as.factor(data$InternetService)
  data$OnlineSecurity<-as.factor(data$OnlineSecurity)
  data$DeviceProtection<-as.factor(data$DeviceProtection)
  data$OnlineBackup<-as.factor(data$gender)
  data$TechSupport<-as.factor(data$TechSupport)
  data$SeniorCitizen<-as.factor(data$SeniorCitizen)
  data$StreamingTV<-as.factor(data$StreamingTV)
  data$StreamingMovies<-as.factor(data$StreamingMovies)
  data$Contract<-as.factor(data$Contract)
  data$PaperlessBilling<-as.factor(data$PaperlessBilling)
  data$PaymentMethod<-as.factor(data$PaymentMethod)
  data$Churn<-as.factor(data$Churn)
  #Standardization
  data$tenure<-scale(data$tenure)
  data$MonthlyCharges<-scale(data$MonthlyCharges)
  data$TotalCharges<-scale(data$TotalCharges)
  
  assign('data',data,envir = globalenv())
  
}


#Handle Senior's Null values
handle_senior<-function(){
   data$SeniorCitizen[is.na(data$SeniorCitizen)]<-0
   assign('data',data,envir = globalenv())
 }


#SPlit Data into train and Test 
split_data<-function(){
  cols<-c('gender','SeniorCitizen','TechSupport','PaperlessBilling',
         'PaymentMethod','TotalCharges','tenure',
         'OnlineSecurity','Contract','Churn')
  data<-data[,cols]
  #data$Churn<-as.factor(data$Churn)
  inTrain<-createDataPartition(data$Churn,p=.65,list = FALSE)
  train_data<-data[inTrain,]
  test_data<-data[-inTrain,]
  assign('inTrain',inTrain,envir = globalenv())
  assign('data',data,envir = globalenv())
  assign('cols',cols,envir = globalenv())
  assign('train_data',train_data,envir = globalenv())
  assign('test_data',test_data,envir = globalenv())
}


#Logistic Regression Model
log_model<-function(){
  #Logistic Regression
  ctrl <- trainControl(method="repeatedcv",repeats = 3)
  log_ <- train(Churn~.,data=train_data,method='glm',family='binomial',
                     preProcess = c("center","scale"),trControl = ctrl,tuneLength=10)
  # Predicting the Test set results
  log_pred<-predict(log_,newdata=test_data)
  log_cm <- confusionMatrix(log_pred, test_data$Churn)$table
  log_accuracy <- (mean(log_pred == test_data$Churn))*100
  #print(paste('Logistic Regression Accuracy',(log_accuracy)*100))
  #print(log_cm)
  #print(cm)
  assign('log.model',log_,envir = globalenv())
  assign('log_accuracy',log_accuracy,envir = globalenv())
  assign('log_cm',log_cm,envir = globalenv())
}

dt_model<-function(){
  dt<-train(Churn~.,method='rpart',data=train_data)
  #print(dt$finalModel)
  dt_pred<-predict(dt,newdata=test_data)
  dt_cm<-confusionMatrix(dt_pred, test_data$Churn)$table
  dt_accuracy<-(mean(dt_pred == test_data$Churn))*100
  assign('dt_cm',dt_cm,envir = globalenv())
  assign('dt.model',dt,envir = globalenv())
  assign('dt_accuracy',dt_accuracy,envir = globalenv())
}

knn_model<-function(){
  ctrl <- trainControl(method="repeatedcv",repeats = 3) 
  knn_ <- train(Churn ~ ., data = train_data, 
                  method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 10)
  knn_pred <- predict(knn_,newdata = test_data )
  knn_accuracy<-(mean(knn_pred == test_data$Churn))*100
  knn_cm<-confusionMatrix(knn_pred, test_data$Churn )$table
  assign('knn.model',knn_,envir = globalenv())
  assign('knn_accuracy',knn_accuracy,envir = globalenv())
  assign('knn_cm',knn_cm,envir = globalenv())
  
}



#Random Forest Model
rfc_model<-function(){
  control <- trainControl(method='repeatedcv', number=10, repeats=3)
  mtry <- sqrt(ncol(train_data))
  tunegrid <- expand.grid(.mtry=mtry)
  rfc_ <- train(Churn~.,data=train_data, method='rf', metric='Accuracy',
                     tuneGrid=tunegrid,trControl=control)
  rfc_pred <- predict(rfc_,newdata = test_data )
  rfc_cm<-confusionMatrix(rfc_pred, test_data$Churn )$table
  rfc_accuracy<-(mean(rfc_pred == test_data$Churn))*100
  assign('rfc.model',rfc_,envir = globalenv())
  assign('rfc_accuracy',rfc_accuracy,envir = globalenv())
  assign('rfc_cm',rfc_cm,envir = globalenv())
}

#Svm Model
svm_model<-function(){
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  svm_linear <- train(Churn ~., data = train_data, method = "svmLinear",
                      trControl=trctrl,
                      preProcess = c("center", "scale"),
                      tuneLength = 10)
  svm_pred <- predict(svm_linear,newdata = test_data )
  svm_cm<-confusionMatrix(svm_pred, test_data$Churn )$table
  svm_accuracy<-(mean(svm_pred == test_data$Churn))*100
  assign('svm_accuracy',svm_accuracy,envir = globalenv())
  assign('svm.model',svm_linear,envir = globalenv())
  assign('svm_cm',svm_cm,envir = globalenv())
  
}

#Naive Bayes Model
naive_model<-function(){
  nb_<- train(Churn~.,data = train_data,method='nb',trControl=trainControl(method='cv',number=10))
  nb_pred <- predict(nb_,newdata = test_data )
  nb_cm<-confusionMatrix(nb_pred, test_data$Churn )$table
  nb_accuracy<-(mean(nb_pred == test_data$Churn))*100
  assign('nb_accuracy',nb_accuracy,envir = globalenv())
  assign('nb.model',nb_,envir = globalenv())
  assign('nb_cm',nb_cm,envir = globalenv())
}
#New Data to Test
gender<-c('Male','Male','Female');SeniorCitizen<-c(0,1,1);
TechSupport<-c('No','No','Yes');PaperlessBilling<-c('Yes','No','Yes')
Contract<-c('Month-to-month','Month-to-month','Month-to-month')
PaymentMethod<-c('Bank transfer (automatic)','Mailed check','Electronic check');
TotalCharges<-c(1236,520,6547);tenure<-c(18,32,10);
OnlineSecurity<-c('Yes','Yes','No')
newTest<-data.frame(gender,SeniorCitizen,TechSupport,PaperlessBilling,PaymentMethod,
                    TotalCharges,tenure,OnlineSecurity,Contract)
newTest$SeniorCitizen<-as.factor(newTest$SeniorCitizen)
newTest$TotalCharges<-scale(newTest$TotalCharges)
newTest$tenure<-scale(newTest$tenure)
###################################################
read_data()
tenure_handle()
pre_preocessing()
handle_senior()
split_data()
log_model()
naive_model()
knn_model()
svm_model()
rfc_model()
dt_model()
#Predict NewTest Dataframe 
log_new<-predict(log.model,newTest)
naive_new<-predict(naive.model,newTest)
rfc_new<-predict(rfc.model,newTest)
dt_new<-predict(dt.model,newTest)
svm_new<-predict(svm.model,newTest)
knn_new<-predict(knn.model,newTest)

print("Logistic Regression Predict New Values classes as  : ")
print(log_new)

print("Naive Bayes Predict New Values classes as  : ")
print(naive_new)

print("Random Forrest Predict New Values classes as  : ")
print(rfc_new)

print("Decision Tree Predict New Values classes as  : ")
print(dt_new)

print("SVM Predict New Values classes as  : ")
print(svm_new)

print("KNN Predict New Values classes as  : ")
print(knn_new)

