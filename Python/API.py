from Final_Assignment import assignment
from flask import Flask,request
import pandas as pd
app = Flask(__name__)
task=assignment()


@app.route('/')
def training():
    task.read_data()
    task.read_data2()
    task.hanlde_tenure()
    task.preprocessing()
    task.handle_Senior()
    task.split_train_test()
    train_file=open('train.txt','r')
    contents=train_file.read()
    clean=task.after_cleaning()
    notClean=task.before_cleaning()
    return "<h1 style='text-align: center'>Data Before Cleaning  </h1>"+notClean.to_html()+"<h1 style='text-align: center'>Data After Cleaning  </h1>"+ clean.to_html()+'\n'+contents+'\n'+"<h2 style='text-align: center'>Warning:Accuracy and prediction of new Values depend on your selection </h2>"
            



@app.route("/train",methods=['GET','POST'])
def train():
    model_name = request.args.get('name')
    task.models(model_name)
    model_file=open('model.txt','r')
    content_model=model_file.read()
    return "<h1 style='text-align: center'>You have Trained Your Model  </h1>"+content_model


@app.route("/evaluation",methods=['GET','POST'])
def evaluate_model():
    acc,cm,f1=task.evaluate_models()
    accuracy_ar=['Accuracy','Confusion Matrix']
    values=[acc,cm]
    #class_rep_val=[cr]
    acc_df=(pd.DataFrame({'Name':accuracy_ar,'Values':values})).to_html(index=False)
    #class_rep=(pd.DataFrame({'Classification Report':class_rep_val})).to_html(col_space=200)
    #evaluation_file=open('evaluation.txt','r')
    #content_evaluate=evaluation_file.read()
    return "<form style='margin-left:550px;margin-top:100px; text-align:centre' action='http://127.0.0.1:9090/prediction'>\
                <centre>\
                        "+acc_df+"\
                        <h1 style='text-align: center;margin:50px' ></h1>\
                        <input style='padding:15px;font-size:20px;margin-top=40px;text-align:centre' type='submit' value='Predict New Values'>\
                </centre>\
                            </form>\
            "


@app.route("/prediction",methods=['GET','POST'])
def prediction():
    pred_html=open('prediction.txt','r')
    pred_content=pred_html.read()
    return pred_content

@app.route("/get_label",methods=['GET','POST'])
def get_label():
    ge=request.args.get('gender')
    sc=int(request.args.get('SeniorCitizen'))
    tsup=request.args.get('TechSupport')
    pb=request.args.get('PaperlessBilling')
    pm=request.args.get('PaymentMethod')
    totch=float(request.args.get('TotalCharges'))
    tenure=float(request.args.get('tenure'))
    osec=request.args.get('OnlineSecurity')
    con=request.args.get('Contract')
    #label_html=open('label.txt','r')
    #label_content=label_html.read()
    #if  ((str(totch).strip()) or (str(tenure).strip())):
     #   churn='You should not leave empty values'
    #else:
    new_label=task.get_label(gender=ge,SeniorCitizen=sc,tenure=tenure,
                             OnlineSecurity=osec,TechSupport=tsup,
                             Contract=con,PaperlessBilling=pb,PaymentMethod=pm,TotalCharges=totch)
    if (new_label==0):
        churn="<h1 style='text-align: center'>No, he has a high probability not to churn </h1>"
    else:
        churn="<h1 style='text-align: center'>Yes, he has a high probability to churn </h1>"
       
        
    
    
    return "<div><h1 style='text-align: center'>Data You have entered about Customer is Classified AS: </h1>\
<p  style='text-align: center' font-size:500px>"+churn+"</p></div>\
<form  align='center' method='GET' action='http://127.0.0.1:9090/'>\
<h1 align='center'>      &#8595  </h1>\
    <input style='padding:15px;font-size:20px' type='submit' value='Back To Home'>\
    <h3 align='center'> You can back to Home Page to Pick a new model</h3>\
    </form>\
<form  align='center' method='GET' action='http://127.0.0.1:9090/prediction'>\
<h1 align='center'>      &#8595  </h1>\
    <input style='padding:15px;font-size:20px' type='submit' value='Test New Values'>\
        <h3 align='center'> Another New test Values </h3>\
        </form>\""


    
    
if __name__ == '__main__':
	app.run(host="127.0.0.1",port=9090)
    