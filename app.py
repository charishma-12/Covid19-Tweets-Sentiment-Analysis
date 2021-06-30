from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("extracted.csv", encoding="ISO-8859-1")

	X = df['Tweet'].apply(str)
	y = df['label']
	X=X.fillna(0)
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	#Naive Bayes Classifier
	from sklearn.linear_model import LogisticRegression

	clf = LogisticRegression()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)


	if request.method == 'POST':
		message = request.form['message']

		data = [message]
		print(message)
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
