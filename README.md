# phising-site-detection
Aim:

To detect Phishing Websites using Machine Learning Techniques.

INTRODUCTION:
Social engineering attacks have become a common security threat nowadays which is being used to reveal private and confidential information by simply tricking the users without getting detected. The attack is performed to gain sensitive information such as username, passwords, and account number. Phishing or Web spoofing technique is a very common example of social engineering attack. These attacks may appear in many types of communication ways such as SMS, messaging, VOIP, fraud emails. Nowadays, web users have multiple accounts on various different websites including social media, email and accounts for banking so they are more vulnerable to these types of attacks.
In a typical phishing attack the victim is lured by sending a spoofed link which redirects the victim to a fake web page The malicious link is placed on the popular web pages or is sent to the victim via an email. The fake webpage looks exactly similar to the legitimate webpage.
Thus, rather than directing the victim to the legitimate web server it will directed to the attacker server.
The solution of antivirus, firewall and designated software do not fully protect us from web spoofing attack. The application of Secure Socket Layer (SSL) and digital certificate (CA) also does not prevent the web user against such attack. In web spoofing attack, the attacker redirects the request to malicious web server.
Secure browsing techniques sometimes does nothing to protect the users from the phishers that have an idea on how the secure connections work. 
The solution developed here contains series of steps to check characteristics of websites Uniform Resources Locators (URLs). URLs of a phishing webpage generally have some unique characteristics that makes it look different from the URLs of the legitimate web page. 


 


MOTIVATION:
With the increase in amount of people using internet and getting registered on various websites and social media platforms, the attacks on Internet have also increased exponentially. Every year we get to hear various cyber attacks even with some tech giants like Google, Facebook, Microsoft etc and billions of money of web users also gets tricked by attackers by using some techniques such as phishing. So all this has led to a serious concern of building some Protection mechanism that can atleast save the common and innocent web users from getting attacked. So all these events motivated us to build an Anti-Phishing platform or portal that can help common web users to get to know whether they are using a phishing website or a legitimate website. We basically used Machine Learning Algorithms to train our model by using a dataset from PhishTank website. After our model got trained it eventually got ready for testing purpose and started to tell the user whether the given website is a phishing website or a legitimate one. So in this way it is very helpful for the Web users as they are now easily protected from the phishing attacks on them done by the phishers by sending some malicious links to steal their personal and financial information and eventually saved a large amount of their money from being getting stolen.

LITERATURE SURVEY:

Background:
A. Blacklist method
This is the most easy and generally used technique in which a list of phishing URLs are stored in some database and on testing a new URL if that is found in the database, it is known as a phishing URL and gives a warning to the user otherwise it is called legitimate website.

B. Heuristic based method
This is an extension of blacklist method and is capable to detect any new phishing attack used by a phisher by analysing features extracted from the phishing site. But the limitation of this method is that it cannot detect all new attacks and easiest to bypass once attacker gets to know the algorithm or features used.

C. Visual similarity
This technique extracts the image from the legitimate website and compares it with the currently visited website. But limitation of this approach is image comparison takes more time as well as more space to store image thus increasing the time and space complexity of this technique. It produces high false negative rate and fails to detect when visual appearance of websites slightly changes.

D. Machine learning
This technique works efficiently in large datasets containing a large amount of data. This also removes drawback of the above mentioned approaches and is capable to detect attacks easily.  Machine Learning based classifiers are efficient classifiers which achieved accuracy more than 99%. Performance of these classifiers depends on size of training and testing data, feature set, and type of classifiers used. 



Proposed Approach:

•	Creating TF-IDF vectorizer for creating compatibility with algorithms for detecting phishing websites.
•	Implementation of TF-IDF algorithm, Logistic Regression, Naïve Bayesian algorithm for training and classifying data.
•	Create back-end using FLASK to call the model for feature engineering and calling the saved models.
•	Develop a web page to reflect the current activity for the phishing website and its output for benign and malicious url.

Logistic Regression: 
Logistic regression is a statistical approach for analysing a dataset in which one or more independent variables are involved in determining an outcome. The outcome is measured with a variable that have only two possible outcomes. This approach is used to predict a binary outcome given a set of independent variables. For representing binary / categorical outcome, dummy variables are being used. We can also think about logistic regression as a special case of linear regression where the outcome variable being involved is categorical, in which we use log of odds as dependent variable. In simple words, it predicts the probability of occurrence of an event by fitting data into a logit function.
Logistic regression was developed by a very popular statistician David Cox around 1958. This binary logistic approach is being used to make an estimation of the probability of a binary response based on one or more predictor variables or features. This model allows us to conclude that the presence of a risk factor increases the probability of a given outcome by a specific percentage.
Like all other regression analysis, the logistic regression is one of the most popular predictive analysis. This approach is used to explain data and to describe the relationship between one dependent binary variable and one or more ordinal, nominal, ratio-level or interval independent variables.


Naive Bayes Classifier:
Naive Bayes classifiers are a family of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a collection of algorithms where each of them share a common principle, i.e. each pair of features being classified is independent of each other.
The dataset is classified into two parts, named as feature matrix and the response vector.
•	Feature matrix have all the rows of the dataset in which each row stores the value of dependent features. 
•	Response vector stores the value of output variable for each row of feature matrix. 
The fundamental Naive Bayes assumption is that each feature makes an independent and equal contribution to the outcome.
Bayes’ Theorem

Bayes’ Theorem finds out the probability of an occurring event given the probability of another event that has already been occurred. Bayes’ theorem is stated mathematically in the equation given below:
 
where A and B are events and P(B) ? 0.

Some of the other popular Naive Bayes classifiers are:
•	Multinomial Naive Bayes: Feature vectors show the frequencies of certain events that have been generated by a multinomial distribution. This model is generally used for document classification.
•	Bernoulli Naive Bayes: In the multivariate Bernoulli model consisting of events, features are independent boolean variables representing inputs. Similar to the multinomial model, this model is also used for document classification purpose, where binary term occurrence (which means whether a word occurs in a document) features are used instead of the term frequencies.





Algorithms Used:

•	Data Preprocessing
•	Creating TF-IDF vectorizer for computation of the dataset.
•	Classifying the data for training and testing purposes.
•	Predicting the accuracy following the result of the input.
•	Developing the frontend using Html, CSS, JavaScript. 
•	Connect this frontend with the backend using FLASK for displaying the output.
•	The resulting output is displayed on the screen in the same page.


.
SYSTEM SPECIFICATIONS:
The following are the main software and hardware requirements:
•	Processor: Intel(R) Core (™) i3/i5/i7
•	Operating System: Windows / Linux / Unix
•	Installed Memory (RAM): minimum 4 GB
•	IDE: Python IDLE, sublime text, Notepad++.
•	Programming Language: Python 

Our prototype is implemented in Python language and uses a machine learning paradigm. It contains models to get trained and produce the correct output. The datasets in use are downloaded from phish tank and then fed to the classifier model to train the system. This system requires the knowledge of Python and machine learning for implementation. Certain software’s such as sublime text editor, Python interpreter, various packages like pandas, sklearn, flask etc. are also required to support various kind of computational tasks.  




IMPLEMENTATION DETAILS:

 
We are using python language to implement our project.Hence we need to import the necessary libraries so that we make our project work.These are pandas which are used to make the files supportive to python and  then import numpy which are used to do numeric calculation under python.Also we require to import random module as well which are to compute random state in the program.These random functions draws only random values according to the compiler wish without any biased values.Hence These python file are used directly as entire module so that it may support our program.

 
Next comes the scikit learn module which is a very powerful tool in python used to execute various algorithms and logics. First comes the Feature extraction module from scikit learn where we import the Term Frequency Inverse Document Frequency which is used to count the specific terms related to the document. It is known as Tfidf Vectorizer as it is used to convert in form of vector.
Next comes the libraries used for model the dataset.Here we used to linearise our model and imported the logistic regression for features linearly related.Afterwards,we import the  necessary model for feature selection and inside it is the splitting training and testing dataset which splits our dataset into two portion -first used to train our dataset and second to test our dataset.

 
Then comes the server used to communicate between the frontend and backend. We used Flask as our embedded server. It is used to connect between client and server. We import several libraries flask which are flask itself in python so that it supports our program and render template which is used to show template in frontend we made in html, css. The url for is used to redirect the pages from one page to other. The request is used to request the client for specific purposes and expect some response.
To run our program we have to first call Flask with the __name__ parameter which is running the flask in the embedded server.
 
Here comes our first function of python where we redirect our user to the home page which is also the root url of our website, Hence ‘/’ and ‘/home’ are two url which is used with the help of decorator. The route function tells python to link the function with the specified url so that it is redirected to that function and do that specific work written on it.
The home_index function just render the template i.e. Home_index html page to the monitor screen using flask.

 
Another function handles the exception in our website. If any error is encountered by user then it is redirected to this function. Hence the link 404 is redirected here to this function which renders the frontend template Exception_page.html and displays 404 error.

 
Next comes the predict function which is the main function to predict whether our site is a phishing site or not. Here it is link to ‘/predict’ link after the root link. We used the parameter in route function here which is the Option field. We can write the different options according to us. here we written whether the data receiving us is get or post and if it is post then only this function will work.
Here we read our dataset which is the comma separated value (csv) file. It is our dataset which contains malicious as well as non malicious sites which is to detect after our machine learning algorithm whether our site is valid or not inputted by the user. It is converted to a data frame as urls_data_sets.Then we segregate them into two values namely the label and url list. The label list store the labels of our dataset and url list stores the urls of our dataset. These are also data frames of python. 


 

Next we are computing Tfidf that is the Term Frequency Term Inverse Document Frequency used to calculate the weight of a particular word in the specific passage. Here we create the vector of there. Tfidf frequencies score and then fit that vector data which is stored in url_data. It is the normalized values of our dataset.
Next we used our random function which is used to randomly split our dataset into training and testing dataset. Size=0.2 splits the dataset into 80% training and 20% testing dataset. Hence there are two training sets of label and url and similarly two for testing label and url dataset. 

 
Next comes our classifier used to predict our program result i.e. MultinomialNB  that stands for Multinomial Naive Bayes which is used to calculate the Bayesian values for multi-dimensional values. It is the classifier used in our program which is predicting the probability of the dataset and calculated values according to it.
Then it fits our data and evaluates its scores using the training and testing dataset.. 

 

We first check whether the method of request by the user is get or post. If it is post then only we transform our phishing website inputted by the user and extract feature in it. We convert these features into a list and then into array, These forms url input array to be given to predict function. We call the predict function here and whatever the result is obtained is rendered into the template of result.html which is our result web page for showing the output of our program.


 



At the very last we run our main function which is the starting line of our program.it checks whether it is the main function and then runs the flask serve with activation of debugger mode which checks the active errors and debugs them. Hence with this our application program runs with the connectivity of frontend ,backend and our flask program together and hence displays the output.
