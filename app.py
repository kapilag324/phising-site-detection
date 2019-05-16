from flask import Flask,render_template,url_for,request
#from flask_bootstrap import Bootstrap

import pandas as pda
import numpy as nup
import random



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
#Bootstrap(app)

@app.route('/')
@app.route('/home')
def Home():
	return render_template('Home.html')

@app.route('/about')
def About():
	return render_template('About.html')

@app.route('/contact')
def Contact():
	return render_template('Contact.html')

@app.errorhandler(404)
def page_not_found(e):
	return render_template('404.html'),404

@app.route('/predict',methods = ['POST'])
def predict():

	urls_data = pda.read_csv('urldata.csv')
	y = urls_data['label']
	url_list = urls_data['url']
	vectorizer = TfidfVectorizer()
	x = vectorizer.fit_transform(url_list)
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
	#Logit = LogisticRegression()
	#Logit.fit(x_train,y_train)

	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(x_train,y_train)
	clf.score(x_test,y_test)


	if request.method == 'POST':
		phish_site = request.form['url'] 
		input_data = [phish_site]
		input_vect = vectorizer.transform(input_data).toarray()
		my_prediction = clf.predict(input_vect)

	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
        app.run(debug = True)
