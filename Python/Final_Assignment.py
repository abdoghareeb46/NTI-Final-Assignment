import pandas as pd
import numpy as np
#import seaborn as sns
import warnings
#import matplotlib.pyplot as plt
#from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', DeprecationWarning)


class assignment:
    
       
    def read_data(self):
        self.data=pd.read_csv('dataset.csv')
    
    
    def before_cleaning(self):
       not_clean=pd.read_csv('dataset.csv')
       not_cleaned=not_clean.head()
       return not_cleaned
        

    def read_data2(self):
        self.data.drop('customerID',axis=1,inplace=True)
        
    def show_data(self):
        print(self.data.head())
        return self.data.head()
    
    def show_info(self):
        print(self.data.info())
        
    def show_describtion(self):
        print(self.data.describe())
    
    def check_nulls(self):
        print('Number Of null values in each Column :','\n\n',self.data.isnull().sum(),'\n\n\n')
    
    def hanlde_tenure(self):
        ten_mean=self.data['tenure'].mean()
        self.data['tenure']=self.data['tenure'].fillna(value=ten_mean)
        #print('\n','Check if Tenure still has null values ','\n\n')
        #self.check_nulls()
        #return self.data

    def preprocessing(self):
            # Label Encoding
            encs = {}
            for col in self.data.columns:
                if self.data[col].dtype == "object":
                    encs[col] = LabelEncoder()
                    self.data[col]   =encs[col].fit_transform(self.data[col])
            # Standardization
            sc=StandardScaler()
            self.data['tenure']=sc.fit_transform(self.data['tenure'].values.reshape(-1,1))
            self.data['MonthlyCharges']=sc.fit_transform(self.data['MonthlyCharges'].values.reshape(-1,1))
            self.data['TotalCharges']=sc.fit_transform(self.data['TotalCharges'].values.reshape(-1,1))
            #print('\n','Data After PreProcessed:......','\n\n')
            #self.show_data()
            #return self.data
            
    #  Handling Missing values in SeniorCitizen by Choosing Best classifier to predict it's values
    def handle_Senior(self):
        df=self.data.copy()
        new_test=df[df['SeniorCitizen'].isnull()]
        new_test.drop('SeniorCitizen',axis=1,inplace=True)
        new_test=new_test[['gender','TechSupport','Churn','PaperlessBilling',
                           'PaymentMethod','TotalCharges','tenure',
                           'OnlineSecurity','Contract']]
        df.dropna(inplace=True)
        
        X=df[['gender','TechSupport','PaperlessBilling','Churn','PaymentMethod',
              'TotalCharges','tenure','OnlineSecurity','Contract']].values
              
        y=df['SeniorCitizen'].values
        
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33, 
                                                         random_state=42)

        log_model=LogisticRegression().fit(X_train,y_train)
        log_pred=log_model.predict(X_test)
        #print('Accuracy Score for KNN While Handling SeniorCitizen Nulls:',
         #     '\n',round(accuracy_score(y_test,knn_pred),3)*100,'%')
        new_senior=log_model.predict(new_test)
        new_test['SeniorCitizen']=new_senior
        self.data['SeniorCitizen']=self.data['SeniorCitizen'].fillna(new_test['SeniorCitizen'])
        #print('\n','Data After Handling Nulls in SeniorCitizen:......','\n\n')
        #self.show_data()
        #return self.data
    def after_cleaning(self):
        cleaned=self.data.head()
        
        return cleaned
        
        return data
    def split_train_test(self):
       X=self.data[['gender','SeniorCitizen','TechSupport','PaperlessBilling',
                           'PaymentMethod','TotalCharges','tenure',
                           'OnlineSecurity','Contract']]
       y=self.data['Churn']
       
       self.X_train,self.X_test, self.y_train,self.y_test = train_test_split(X,y, 
                                                test_size=0.33, random_state=42)
    
    def models(self,model_name):
        if model_name=='LogisticRegression':
            self.model=LogisticRegression(C=0.04,max_iter=999999999)
        elif(model_name=='KNN'):
            self.model = KNeighborsClassifier(n_neighbors=14)
        elif(model_name=='RandomForest'):
            self.model=RandomForestClassifier(max_depth=None,max_features='log2',
                                              min_samples_leaf=20,n_estimators=100)
        elif(model_name=='DecisionTree'):
            self.model = DecisionTreeClassifier()
            
        elif(model_name=='GaussianNB'):
            self.model = GaussianNB()
            
        elif(model_name=='SVM'):
            self.model = SVC(kernel='rbf')
        
        
        self.model.fit(self.X_train,self.y_train)
            
            

            
    def evaluate_models(self):
        
        model_prediction=self.model.predict(self.X_test)
       # print('Accuracy Score for Model You have choosed : ',
        #      round(accuracy_score(self.y_test,model_prediction),3)*100,'%')
        
        accuracy  =  round(accuracy_score(self.y_test,model_prediction),3)*100
        ##print('Confusion Matrix for Model You have choosed: ','....','\n',
          #    confusion_matrix(self.y_test,model_prediction))
        cm=confusion_matrix(self.y_test,model_prediction)
        
        #print('Classification  Report for Model You have choosed: ','....','\n'
         #     ,classification_report(self.y_test,model_prediction))
        f1=f1_score(self.y_test,model_prediction)
        
        return accuracy, cm, f1

    def get_label(self,gender, SeniorCitizen, tenure, OnlineSecurity,TechSupport,Contract,
                  PaperlessBilling, PaymentMethod,TotalCharges):
        
        #Converting categorical data to lower cases to be easy to check for thier values
        gender=gender.lower()
        #SeniorCitizen=SeniorCitizen.lower()
        #Partner=Partner.lower()
        #Dependents=Dependents.lower()
        #PhoneService=PhoneService.lower()
        #MultipleLines=MultipleLines.lower()
        #InternetService=InternetService.lower()
        OnlineSecurity=OnlineSecurity.lower()
        #OnlineBackup=OnlineBackup.lower()
        #DeviceProtection=DeviceProtection.lower()
        TechSupport=TechSupport.lower()
        #StreamingTV=StreamingTV.lower()
        #StreamingMovies=StreamingMovies.lower()
        Contract=Contract.lower()
        PaperlessBilling=PaperlessBilling.lower()
        PaymentMethod=PaymentMethod.lower()
        #MonthlyCharges=MonthlyCharges
        TotalCharges=TotalCharges
        tenure=tenure
        
        #Check gender value
        if gender=='male':
            ge=1;
        elif gender== 'female':
            ge=0;
            
            #check for SeniorCitixen value
        if SeniorCitizen==0:
            sc=0;
        elif SeniorCitizen== 1:
            sc=1;
            
            
        #Check Partner value    
        #if Partner== 'yes':
         #   pt=1;
        #elif Partner=='no':
        #    pt=0;
            
            #Check Dependents value
        #if Dependents== 'yes':
         #   dp=1;
        #elif Dependents=='no':
         #   dp=0;
  
            #Check PhoneService value
        #if PhoneService=='Yes' or 'yes':
         #    ps=1;
        #elif PhoneService=='No' or 'no':
         #    ps=0;
            
            #Check MultipleLines value
        #if MultipleLines=='Yes' or 'yes':
         #    ml=2;
        #elif MultipleLines=='No' or 'no':
         #    ml=0;
        #elif MultipleLines=='No phone service':
         #    ml=1;
            
            #Check InternetService value
        #if InternetService== 'dsl':
         #    iser=0;
        #elif InternetService=='fiber optic':
         #    iser=1;
        #elif InternetService=='no':
         #    iser=2;
             
             #Check OnlineSecurity value
        if OnlineSecurity== 'no':
             osec=0;
        elif OnlineSecurity=='no internet service':
             osec=1;
        elif OnlineSecurity=='yes':
             osec=2;
            
            #Check OnlineBackup value
        #if OnlineBackup== 'no':
         #    obac=0;
        #elif OnlineBackup=='no internet service':
         #    obac=1;
        #elif OnlineBackup=='yes':
         #    obac=2;
             
             #Check DeviceProtection value
        #if DeviceProtection== 'no':
         #    dpro=0;
        #elif DeviceProtection=='no internet service':
         #    dpro=1;
        #elif DeviceProtection=='yes':
         #    dpro=2;
             
             
             #Check TechSupport value
        if TechSupport== 'no':
             tsup=0;
        elif TechSupport=='no internet service':
             tsup=1;
        elif TechSupport=='yes':
             tsup=2;
             
             #Check StreamingTV value
        #if StreamingTV== 'no':
         #    sttv=0;
        #elif StreamingTV=='no internet service':
         #    sttv=1;
        #elif StreamingTV=='yes':
         #    sttv=2;
             
             #Check StreamingMovies value
        #if StreamingMovies== 'no':
         #    stmo=0;
        #elif StreamingMovies=='no internet service':
         #    stmo=1;
        #elif StreamingMovies=='yes':
         #    stmo=2;
             
             #Check Contract value
        if Contract== 'month-to-month':
             con=0;
        elif Contract=='one year':
             con=1;
        elif Contract=='two year':
             con=2;
             
             #Check PaperlessBilling value
        if PaperlessBilling== 'no':
             pb=0;
        elif PaperlessBilling=='yes':
             pb=1;
        
            #Check PaymentMethod value
        if PaymentMethod== 'bank transfer (automatic)':
             pm=0;
        elif PaymentMethod=='credit card (automatic)':
             pm=1;
        elif PaymentMethod=='electronic check':
             pm=2;
        elif PaymentMethod=='mailed check':
             pm=3;
        
        predicted_label=self.model.predict(np.array([ge,sc,tsup,pb,pm,
                                                     TotalCharges,tenure,osec,con]).reshape(1,-1))[0]
        return predicted_label
        #print ('Churn Class for entered data is : ',
         #      self.model.predict(np.array([ge,sc,pt,dp,tenure,ps,ml,iser,osec,obac,dpro,tsup,sttv,stmo,con,
          #                                  pb,pm,MonthlyCharges,TotalCharges]).reshape(1,-1)))
    
            
            
if __name__=='__main__':
    final=assignment()
    final.read_data()
    final.read_data2()
    final.show_data()
    #final.show_info()
    #final.show_describtion()
    #final.check_nulls()
    #final.hanlde_tenure()
    #final.preprocessing()
    #final.handle_Senior()
    #final.split_train_test()
    #final.models('SVM')
    #acc,cmm,crr=final.evaluate_models()
    #print("Accuracy is: ",acc,"\n ",'Confusion Matrix is: ',cmm
     #     ,'\n ','Classification Reprort IS','\n',crr)
    #final.get_label('female') #Enter your desired data to predict it's new class
    