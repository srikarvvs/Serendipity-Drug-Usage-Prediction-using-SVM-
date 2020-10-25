from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from nltk import ngrams
import numpy as np
import textblob
import re, nltk
from nltk.stem import WordNetLemmatizer
import nltk
import collections
from nltk.corpus import stopwords
#directly downloads packages for required things
#run this following two lines only on first execution
nltk.download('punkt')
nltk.download('popular')
from nltk.corpus import stopwords
import  pandas as pd
import string
#importing the data
from collections import Counter
from textblob.sentiments import PatternAnalyzer
import pickle
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk,Image
import operator
from itertools import islice
import matplotlib.ticker as ticker



wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def take(n, iterable):
	return list(islice(iterable, n))

def read_data():
	try:
		reading_the_csv=csvfile 
		DRUGNAME=amount.get()
		if len(DRUGNAME)!=0:
			try:
				raw_data = csvfile
				#raw_data['drugName']= raw_data['drugName'].str.lower()
				#raw_data_filtered=raw_data[raw_data['drugName']!= DRUGNAME]
				return raw_data
			except: 
				messagebox.showinfo("Error", "File Error")
			
		else:
			messagebox.showinfo("Error", "Enter The Drug Name")
	except:
		messagebox.showinfo("Error", "File Error")

	#return raw_data
def preprocess_drug():
	DRUGNAME=amount.get()

	if len(DRUGNAME)>0:
		try:
			f=open('svm.pkl','rb');
		except:
			messagebox.showinfo("Error", "Model Not Found")
		try:
			raw_data = csvfile
		except:
			messagebox.showinfo("Error", "File error")
		#raw_data["drugName"]=raw_data["drugName"].lower()
		raw_data_filtered=raw_data[raw_data['drugName']== DRUGNAME]
		if len(raw_data_filtered)>0:
			print("drug specified length")
			print(len(raw_data_filtered))
			raw_data_filtered["normalized_reviews"]=np.nan
			print("[INFO] Normalizing...")
			raw_data_filtered["normalized_reviews"]=raw_data_filtered["reviews"].apply(lambda x: normalizer(x))
			print("[INFO] Normalized ")
			raw_data_filtered["processed_reviews"]=np.nan
			raw_data_filtered["sentiment"]=np.nan
			raw_data_filtered["processed_reviews"]=raw_data_filtered["normalized_reviews"].apply(lambda x: " ".join(list(x)))
		    #print(test_data["processed_reviews"].head())
			print("[INFO] Getting Polarity Value")
			raw_data_filtered["sentiment"]=raw_data_filtered["processed_reviews"].apply(lambda x: sentiment(x))


			raw_data_filtered.to_csv("processed_reviews2.csv")
			

			X= raw_data_filtered[['sentiment']]
		# normalize_review=normalizer(tweet)
		# 	processed_reviews=" ".join(list(normalize_review))
		# 	sent=sentiment(processed_reviews)	
			
			clf=pickle.load(f);
			f.close()
			Y=clf.predict(X)
			dft1 = pd.DataFrame()
			dft1 = X
			dft1['predictions'] = list(Y)
			dft1['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
		                                                            title='specific prediction distrubution ',
		                                                            color=['red','blue','green'])
			plt.xlabel("PREDICTED CLASSES")
			plt.ylabel('COUNT')
			plt.show()
		else:
			messagebox.showinfo("Error", "Enter The Valid drug name")
	else:
		messagebox.showinfo("Error", "Enter The Drug Name")
    
def normalizer(tweet):
    alphabets = re.sub("[^a-zA-Z]", " ",tweet)
    tokens = nltk.word_tokenize(alphabets)
    lower_case = [l.lower() for l in tokens]
    processed_words = list(filter(lambda l: l not in stop_words, lower_case))
    #print(processed_words)
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in processed_words]
    return lemmas

def sentiment(tweet):
    #return textblob.TextBlob(tweet).sentiment.polarity
    sentiment_analyzer = PatternAnalyzer()
    return sentiment_analyzer.analyze(tweet).polarity
def feauture_labelling(score):
    if (score >= 0.1):
        return 'positive'

    elif ((score > -0.1) and (score < 0.1)):
        return 'neutral'

    elif ((score <= -0.1) ):
        return 'negative'


def process():
	conditions=condition1.get()
	data=pd.read_csv(r'preprocessed_data.csv', engine='python')
	data=data[data['condition']== conditions]
	if len(data)==0:
		messagebox.showinfo("Error", "Enter The Condition correctly")
	else:

		print(len(data))
		name_of_drug1=data.drugName.unique()
		name_of_drug=list(name_of_drug1)
		#print(name_of_drug)
		dict1={}
		dict2={}
		f=open('svm.pkl','rb');
		clf=pickle.load(f);
		f.close()
		column_names=['drugName','condition','sentiment','score']
		udf = []
		try:
			for i in name_of_drug:
				data1=data[data['drugName']== i]
				#print(data1)
				if(len(data1)>50):
					dict1[i]=len(data1)
					data2=data1[['drugName','condition','sentiment']]
					data3=data1[['sentiment']]
					y_pred = clf.predict(data3)
					data2['score']=list(y_pred)
					positivity=data2[data2['score']=='positive'] 
					# negativity=data2[data2['score']=='negative']
					# neutral=data2[data2['score']=='neutral']
					positive_rate=(len(positivity)/len(data1))*100
					dict2[i]=positive_rate
					udf.append(data2)
			updf=pd.concat(udf)

			#print(updf)
			# #dict containing the all the possible drugs of given condition
			# print("dict containing the all the possible drugs of given condition")
			# print(dict2)
			#dict containing the top5 list in descending order
			print("dict containing the top5 list in descending orde")
			sorted_dict=dict(sorted(dict2.items(), key=operator.itemgetter(1),reverse=True))
			print(sorted_dict)
			if(len(sorted_dict)>=5):
				n_items = take(5, sorted_dict.items())
			else:
				n_items = take(len(sorted_dict), sorted_dict.items())
			print(n_items)
			top5list=[]
			for i in n_items:
				top5list.append(i[0])
			print(top5list)
			data5=updf[updf['drugName'].isin(top5list)]
			airline_sentiment = data5.groupby(['drugName','score']).score.count().unstack()
			print(airline_sentiment.head())
			sentimentvalues=['negative','neutral','positive']
			from matplotlib import ticker as mt
			for s in sentimentvalues:
				airline_sentiment[s] = airline_sentiment[s].astype(float)
			for i in top5list:
				s=0
				for j in sentimentvalues:
					s+=airline_sentiment[j][i]
				for j in sentimentvalues:
					airline_sentiment[j][i]=round((airline_sentiment[j][i]/s)*100,2)
			print(airline_sentiment.head())
			ab = airline_sentiment.plot(kind='bar')
			ab.yaxis.set_major_formatter(mt.PercentFormatter())
			ab.set_ylim(0,100)
			for p in ab.patches:
				ab.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
			plt.xlabel("Drug Names")
			plt.ylabel('Percentage')
			plt.show()
		except Exception as e:
			print("Reviews donot satisy the required condition(",e,")")

SAVE_DATA_PATH = "preprocessed_data.csv"
def preprocess(test_data):
    # pass the test dataset here
    #print("given size of the data", len(test_data))
    # dropping the duplicate values
    test_data = test_data.drop_duplicates()
    print("[INFO] Duplicates Removed")
    #drop Rows/Columns with Null values
    test_data =test_data.dropna(axis=0, how='any')
    print("[INFO] Missing Values Removed")
    #print(" size of data after removing repitations", len(test_data))
    #print(test_data['TWEETS'].value_counts())
    # DATA PRE  PROCESSING STARTS FROM HERRE
    # create a coloum in the dataset to get only normalizered words
    test_data["normalized_reviews"] = np.nan
    print("[INFO] Normalizing...")
    test_data["normalized_reviews"] = test_data["reviews"].apply(lambda x: normalizer(x))
    print("[INFO] Normalized ")
    # now we are converting to ngrams to get the meanining
    test_data["processed_reviews"] = np.nan
    test_data["sentiment"] = np.nan
    test_data["processed_reviews"] = test_data["normalized_reviews"].apply(lambda x: " ".join(list(x)))
    #print(test_data["processed_reviews"].head())
    print("[INFO] Getting Polarity Value")
    test_data["sentiment"] = test_data["processed_reviews"].apply(lambda x: sentiment(x))
    
    #print(test_data["sentiment"].value_counts())

    test_data["score"] = test_data["sentiment"].apply(lambda x: feauture_labelling(x))
    print("[INFO] Completed ")
    test_data.to_csv(SAVE_DATA_PATH)
    return test_data
def naivebayes(test_data):

    from sklearn.model_selection import train_test_split
    test_data= test_data[['sentiment', 'score']]

    #To remove the SetCopy Warning 
    pd.set_option('mode.chained_assignment',None)

    X = test_data.drop(columns=['score'])
    print("\n ********dimension of input trained data*******\n ", X.shape)
    Y = test_data['score'].astype("str")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

    from sklearn.naive_bayes import GaussianNB
    # Create a nb Classifier
    clf = GaussianNB() # f=open('naivebayes.pkl','rb');clf=pickle.load(f);f.close()
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    

    from sklearn import metrics

    # Model Accuracy: how often is the classifier correct?

    dft = pd.DataFrame()
    dft = x_test
    dft['predictions'] = list(y_pred)

    print("Accuracy  for Naive Bayes :", metrics.accuracy_score(y_test, y_pred)*100)

    

    from sklearn.metrics import confusion_matrix
    # cm=confusion_matrix(y_test,y_pred)
    # print(cm)
    # labels=['neutral','negative','positive']
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(cm)
    # plt.title('Confusion matrix of the classifier')
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()

    print("plotting the distrubution in  test split")
    dft['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                            title='NAIVE BAYES  prediction distrubution in test data',
                                                            color=['cyan','teal','pink'])


    plt.xlabel("PREDICTED CLASSES")
    plt.ylabel('COUNT')
    #plt.savefig('svm.png')
    plt.show()

    



def supportvm(test_data):
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    test_data = test_data[['sentiment', 'score']]

    #to eliminate copy warning for csv file
    pd.set_option('mode.chained_assignment',None)


    X = test_data.drop(columns=['score'])
    print("\n ********dimension of input trained data*******\n ", X.shape)
    Y = test_data['score'].astype("str")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

    

    # Create a SVC Classifier
    clf = SVC(kernel='poly')
    # print(x_train.value_counts())
    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    # print(y_pred)
    # model evaluation
    f = open('svm.pkl','wb')
    saved_model=pickle.dump(clf,f)
    f.close()

    from sklearn import metrics

    # Model Accuracy: how often is the classifier correct?

    dft = pd.DataFrame()
    dft = x_test
    dft['predictions'] = list(y_pred)
    print("Accuracy for SVC :", metrics.accuracy_score(y_test, y_pred)*100)
    
    # from sklearn.metrics import confusion_matrix
    # cm=confusion_matrix(y_test,y_pred)
    # print(cm)
    # labels=['neutral','negative','positive']
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(cm)
    # plt.title('Confusion matrix of the classifier')
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()


    

    print("plotting the distrubution in  test split")
    dft['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                        title='SVM  prediction distrubution in test data',
                                                        color=['red','blue','green'])

    plt.xlabel("PREDICTED CLASSES")
    plt.ylabel('COUNT')
    plt.savefig('svm.png')
    plt.show()
    




def getCSV ():
    global csvfile
    
    import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    print("*********")
    csvfile = pd.read_csv (import_file_path)
    print (csvfile) 


def onclick(args):
	if args==1:
		print("Button Clicked")
		print("Loading...")
		DRUGNAME=amount.get()
		import os
		#if os.path.exists(SAVE_DATA_PATH):

		#	p_df = pd.read_csv(SAVE_DATA_PATH)
		#else:
		raw_data=read_data()
		p_df=preprocess(raw_data)


		print("Preprocessing Done")
		#print(p_df.head())
		print("[INFO] Support Vectoring")
		supportvm(p_df)


	if args==2:
		print("Button 2 clicked")
		print("Loading...")
		DRUGNAME=amount.get()
		import os
		#if os.path.exists(SAVE_DATA_PATH):

		#	p_df = pd.read_csv(SAVE_DATA_PATH)
		#else:
		raw_data=read_data()
		p_df=preprocess(raw_data)


		print("Preprocessing Done")
		#print(p_df.head())
		print("[INFO] Naive Bayes")
		naivebayes(p_df)
def create_window():
    window = tk.Toplevel(root)
    width,height=window.winfo_screenwidth(),window.winfo_screenheight()
    window.geometry('%dx%d+0+0' %(width,height))
    window.title("Condition Based")
    title1=tk.Label(window,text="Predicting Top 5 Drugs on positive rate for given condition",fg='red',font=('helvetica', 35, 'bold')).place(x=100,y=50)

    cond=tk.Label(window,text="Enter the Condition ",font=('helvetica', 18, 'bold')).place(x=450,y=150)
    global condition1
    condition1=tk.StringVar()
    e1=tk.Entry(window,textvariable=condition1,font=('helvetica', 18, 'bold')).place(x=700,y=150)
    upload = tk.Button(window, text="Top 5 List ",command=process,bg='green', fg='white', font=('helvetica', 18, 'bold')).place(x=700,y=250)


root=tk.Tk()
width,height=root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry('%dx%d+0+0' %(width,height))
root.title("Main Page")

#Enter the Drugname
title1=tk.Label(root,text="Serendipity Drug Usage Prediction",fg='red',font=('helvetica', 35, 'bold')).place(x=400,y=50)
drugname=tk.Label(root,text="Enter the Drug Name",font=('helvetica', 18, 'bold')).place(x=450,y=150)
amount =tk.StringVar()
e1=tk.Entry(root,textvariable=amount,font=('helvetica', 18, 'bold')).place(x=700,y=150)

upload = tk.Button(root, text="Upload",command=getCSV,bg='green', fg='white', font=('helvetica', 18, 'bold')).place(x=700,y=250)
#Creating Buttons
btn1 = tk.Button(root, text="SVM & Graph",command=lambda:onclick(1),bg='green', fg='white', font=('helvetica', 18, 'bold')).place(x=660,y=350)
btn2 = tk.Button(root, text="Naive Bayes & Graph",command=lambda:onclick(2),bg='green', fg='white', font=('helvetica', 18, 'bold')).place(x=625,y=450)
btn3 = tk.Button(root, text="Specific Drug and graph",command=preprocess_drug,bg='green', fg='white', font=('helvetica', 18, 'bold')).place(x=615,y=550)
next1 = tk.Button(root, text="Predicting Top 5 Drugs on given condition", command=create_window,bg='green', fg='white', font=('helvetica', 18, 'bold')).place(x=550,y=650)
exit = tk.Button(root, text="Exit",command=root.quit,bg='red', fg='black', font=('helvetica', 16, 'bold')).place(x=720,y=750)


root.mainloop()
