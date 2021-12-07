from flask import Flask,request,url_for,redirect,render_template  #importting necessary librabries
import pickle #pickle will load our output file from notebook 
import pandas as pd  #fror converting data in to dataframe for input


app = Flask(__name__) #flask name

pred_model = pickle.load(open("Diabetes_pred.pkl" , "rb"))   #Loading our prediction model

@app.route('/')   #main route
def home():
	return render_template("index.html")    #homepage


@app.route('/predict',methods= ['POST','GET'])    #POST AND GET Methods will be called on clicking predict button 
def predict():
	# we have 8 fields for prediting outcome therefore fetching each data field one by one

	data1 = request.form['1']
	data2 = request.form['2']
	data3 = request.form['3']
	data4 = request.form['4']
	data5 = request.form['5']
	data6 = request.form['6']
	data7 = request.form['7']
	data8 = request.form['8']
	

	#printing row
	
	data_df = pd.DataFrame([pd.Series([data1,data2,data3,data4,data5,data6,data7,data8])])  #Creation of DataFrame using all the data

	print(data_df)

	prediction_diabetes = pred_model.predict_proba(data_df)   #Output Prediction

	model_output = '{0:.{1}f}'.format(prediction_diabetes[0][1], 2)    #formatting the outcome


	if model_output>str(0.5):
		return render_template('index.html', pred = 'You have chance of having Diabetes, Please visit Doctor. \n Probability of Diabetes is: {} '.format(model_output))
		return render_template('prediction.html', pred = 'You have chance of having Diabetes, Please visit Doctor. \n Probability of Diabetes is: {model_output}')

	else:
		return render_template('index.html', pred = 'You are safe. \n Probability of having Diabetes: {}'.format(model_output))
		return render_template('prediction.html', pred = 'You are safe. \n Probability of having Diabetes: {model_output}')


if __name__ == '__main__':
	app.run(debug=True)
