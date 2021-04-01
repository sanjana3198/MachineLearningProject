#pip install keras
#pip install tensorflow


############ Restaurant Review Classification System #############

#The data preprocessing step is crucial part of the Data Science spectrum.
#Depending upon the type of data, the preprocessing steps vary.

#Continuous or Categorical --------> Data Preprocessing
#Images or Videos -----------------> Image Preprocessing
#Textual Data ---------------------> Natural Language Processing

#Data Preprocessing is a broad step which is subjective to the problem we have solved and the data we have.

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read dataset reviews.tsv
dataset = pd.read_csv('reviews.tsv', sep = '\t')

#By inspecting the data, we understand that it is a typical binary classification problem
#wherein we have to predict sentiments associated with a particular review.

#Positive Review: 1
#Negative Review: 0

#The only problem is that the input data (feature matrix) is not continuous or categorical in nature.
#It is actually textual.
#In order to preprocess the data so that a machine learning algorithm can make sense out of it,
#we need to apply certain NLP techniques.

#The general steps to preprocess textual data

#1. Remove all the numbers, punctuations, emojis and unwanted characters.
#2. Getting all the data into a similar case (lower case).
#3. Remove all the unwanted words like preprositions, conjunctions, determiners, pronouns and fillers etc.
#4. Perform stemming or lemmatization.
#5. Select an nlp model to represent the data (Bag or Words, TF-IDF Word Vector etc).

## Test Review to understand all the preprocessing steps ##
    
#"The food here was totally delicious!! *Smiling emoji*. 100% \
#recommended"

#1. The food here was totally delicious recommended

#2. the food here was totally delicious recommended

#3. food totally delicious recommended

#Stemming vs Lemmatization

#Stemming trims down a word syntax wise to return the root word
#Lemmatization trims down a word semantic wise to return the root word

#Sample three review

#"I love the food here."
#"I'm loving it."
#"I loved the food here."

#In the dataset, for positive predictors we can consider {love, loving, loved} all three
#but a better approach to save space and time complexity would be to consider the root word only.

#{love, loving, loved} ----> lov / love (Depends on the library)

#{Good, Better, Best} ---> Stemming does not makes sense here.
#Lemmatization would derive the root word semantically (meaning wise)

#{Good, Better, Best} ----> good

#4. food total delici recommend

#5. Once preprocessing is done, select the bag of words model to convert textual data into a sparse matrix
  
#Bag of Words in Action

#food : 0
#total : 1
#delici : 2
#recommend : 3

#Note: We have only preprocessed one review and the bag of words model gives 4 unique integer labels.
#Now suppose we have 1 lac reviews, in this case the bag of words model might
#return few lac unique integers for unique words.
    
#Some other review: "Food was extremely tasty"
#After NLP: "food extreme tasty"

#extreme: 4
#tasty: 5

#After all unique words have been assigned unique labels, we create a sparse matrix
#where the columns represent all the words and the rows represent a particular review.

#Columns are all the unique words

#Columns -> 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16
#Row 1 ->   1   1   1   1   0   0   0   0   0   0   0   0   0   0   0   0   0
#Row 2 ->   1   0   0   0   1   1   0   0   0   0   0   0   0   0   0   0   0

#Fill 1 if the review contains the word from the column else put 0

#Now we will actually apply all the aforementioned steps.
#For which we need a special library known as nltk (natural language toolkit)
#We will also use re library that is regular expressions library
import re
import nltk
nltk.download('stopwords')

#stopwords is a module in nltk that contains all the unwanted words
#like he, she, it, when, was, where, the, a, an etc

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#dataset['Review'][0]

#In regular expression: [^a-zA-Z] means catch everything which is not
#in the range of small a to small z and Capital A to Capital Z.
#^ (Carrot symbol) is used for negation.

ps = PorterStemmer()
clean_reviews = []

for i in range(dataset.shape[0]):
    temp = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    temp = temp.lower()
    temp = temp.split()
    #t1 = [word for word in temp if not word in stopwords.words('english')]
    #t2 = [ps.stem(word) for word in temp if not word in stopwords.words('english')]
    temp = [ps.stem(word) for word in temp if not word in stopwords.words('english')]
    temp = ' '.join(temp)
    clean_reviews.append(temp)

# For the BOW Model we will use CountVectorizer class from sklearn
    
from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer()
X = cv.fit_transform(clean_reviews)
X = X.toarray()
y = dataset['Liked'].values

# If you want to know what are these 1565 words which have been made into a column as a sparse matrix
#then you can use this syntax
print(cv.get_feature_names())

#Note: Statistics is the study of general trends and not special
#exceptions. In some cases, the column number could become
#extensively large for you to work. In that case, we only use
#a selected fragment of columns and drop everything else
#by using an argument.

#If you decrease the number of columns then the performance of
#your model would also decrease but speed and efficiency would
#increase. There is a tradeoff that we have to establish
#between column number and model's accuracy.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 7)

#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#Applying KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

#Applying DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()
dtf.fit(X_train, y_train)

#Applying GaussianNB
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

#Applying RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#Cross Validation
from sklearn.model_selection import cross_val_score

cross_val_score(log_reg, X_train, y_train, cv = 5)
cross_val_score(knn, X_train, y_train, cv = 5)
cross_val_score(dtf, X_train, y_train, cv = 5)
cross_val_score(nb, X_train, y_train, cv = 5)
cross_val_score(rf, X_train, y_train, cv = 5)

#cross_val_predict
from sklearn.model_selection import cross_val_predict

y_pred_log = cross_val_predict(log_reg, X_test, y_test, cv = 5)
y_pred_knn = cross_val_predict(knn, X_test, y_test, cv = 5)
y_pred_dtf = cross_val_predict(dtf, X_test, y_test, cv = 5)
y_pred_nb = cross_val_predict(nb, X_test, y_test, cv = 5)
y_pred_rf = cross_val_predict(rf, X_test, y_test, cv = 5)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

#Accuracy Score
print("Logistic Regression : ",accuracy_score(y_test, y_pred_log))
print("KNN : ",accuracy_score(y_test, y_pred_knn))
print("Decision Tree : ",accuracy_score(y_test, y_pred_dtf))
print("Naive Bayes : ",accuracy_score(y_test, y_pred_nb))
print("Random Forest : ",accuracy_score(y_test, y_pred_rf))


#Confusion Matrix
confusion_matrix(y_test, y_pred_log)
# =============================================================================
confusion_matrix(y_test, y_pred_knn)
# =============================================================================
confusion_matrix(y_test, y_pred_dtf)
confusion_matrix(y_test, y_pred_nb)
confusion_matrix(y_test, y_pred_rf)

from sklearn.metrics import auc, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve

#metrics for log_reg
y_log_scores = cross_val_predict(log_reg, X_test, y_test, cv = 5, method = 'decision_function')

print("The ROC AUC Score for Logistic Regression :", roc_auc_score(y_test, y_log_scores))

fpr, tpr, threshold = roc_curve(y_test, y_log_scores)

plt.plot(fpr, tpr, c = "r", label = "Logistic Regression")
plt.plot([0, 1], [0, 1], c = "b")
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.grid(True)
plt.title('ROC Curve')

print("Logistic Regression AUC :", auc(fpr, tpr))

#metrics for rf
y_rf_scores = cross_val_predict(rf, X_test, y_test, cv = 5, method = 'predict_proba')
y_rf_scores = y_rf_scores[:, 1]

print("The ROC AUC Score for Random Forest :", roc_auc_score(y_test, y_rf_scores))

fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, y_rf_scores)

plt.plot(fpr, tpr, c = "r", label = "Logistic Regression")
plt.plot(fpr_rf, tpr_rf, c = "g", label = "Random Forest")
plt.plot([0, 1], [0, 1], c = "b")
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend()
plt.grid(True)
plt.title('ROC Curve')



#Note: If you wanted to create a classification project then
#apart from the aforementioned code, plot a couple of graphs,
#get auc score and you are done. BUT (IN CAPITAL LETTERS)

#Project != Product

#Product has to be deployed somewhere. Popular deployment
#mediums are:
#    1. Cloud (AWS, AZURE, WATSON)
#    2. Browser (Javascript, Django, Flask)
#    3. Mobile Device (Android, IOS)
#    4. IOT or Arduino
#   
#Basically, we want to create a system where user inputs a 
#feedback or review and the system outputs a positive or
#negative symbol.

def run():
    
    print("Please tell us your feedback!")
    str = input()
    #print(str)
    # Performing preprocessing is left for you (hint)
    str = [str] # Changing to list so that Count Vectorizer can work
    #print(str)
    X_str = cv.transform(str) # CV changing your data into sparse matrix
    X_str = X_str.toarray() # Converting to array
    #print(X_str)
    y_str = log_reg.predict(X_str) # Getting prediction
    #print(y_str)
    y_str = y_str[0] # Extracting first element from the list
    
    if y_str == 1:
        print("Logistic Regression says --> Yayy! Positive Review")
    else:
        print("Logistic Regression says --> Negative Review! Our Apologies!")
    
    y_str_knn = knn.predict(X_str) # Getting prediction
    #print(y_str)
    y_str_knn = y_str_knn[0] # Extracting first element from the list
    
    if y_str_knn == 1:
        print("KNN says --> Yayy! Positive Review")
    else:
        print("KNN says --> Negative Review! Our Apologies!")
        
    total_y = (y_str + y_str_knn) / 2
    total_y = np.abs(total_y)
    print(total_y)
    
run()

# Remember to apply preprocessing to new data as well. 
# Just take the loop of preprocessing and put it inside a function
# Call that function from inside of run() function

#In the specialized domain of NLP, a data scientist works with
#characters from different subsets.

#In the specialized domain of Image Processing or Object Detection,
#you work with pixels.

#In the specialized domain of Audio, you work with frequencies.

#"Jack of All. Master of One."

#NLP --> BOW --> TF-IDF --> Similarity --> Embeddings --> RNN --> 
#--> LSTM --> GRU --> Transformers --> BERT --> Robert --> Roberta

#Image Processing --> Grayscaling --> Feature Enhancers --> CNN -->
#--> AlexNet --> ResNet --> GoogleLeNet --> EfficientNet --> YOLO
#--> SSD


















































































































































































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



                                              








































































