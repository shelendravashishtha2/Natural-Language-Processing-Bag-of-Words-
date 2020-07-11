import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
data =pd.read_csv('labeledTrainData.tsv',delimiter='\t',quoting=3)
data=data[0:1000]
data.info()
data.head()
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
stops = stopwords.words('english')
data.info()
import seaborn as sns
sns.set()
data.hist()
stops.remove('don')
stops.remove("don't")
stops.remove('aren')
stops.remove("aren't")
stops.remove('couldn')
stops.remove("couldn't")
stops.remove('didn')
stops.remove("didn't")
stops.remove('doesn')
stops.remove("doesn't")
stops.remove('hadn')
stops.remove("hadn't")
stops.remove('hasn')
stops.remove("hasn't")
stops.remove('haven')
stops.remove("haven't")
stops.remove('isn')
stops.remove("isn't")
stops.remove('mightn')
stops.remove("mightn't")
stops.remove('mustn')
stops.remove("mustn't")
stops.remove('needn')
stops.remove("needn't")
stops.remove('shan')
stops.remove("shan't")
stops.remove('shouldn')
stops.remove("shouldn't")
stops.remove('wasn')
stops.remove("wasn't")
stops.remove('weren')
stops.remove("weren't")
stops.remove('won')
stops.remove("won't")
stops.remove('wouldn')
stops.remove("wouldn't")
stops.remove('no')
stops.remove('nor')
stops.remove('not')
print(stops)
for i in range(1000):
  rev = re.sub('[^a-zA-Z]',' ',data['review'][i])
  rev = rev.lower()
  rev = rev.split()
  ps = PorterStemmer()
  rev = [ps.stem(word) for word in rev if not word in set(stops)]
  rev =' '.join(rev)
  corpus.append(rev)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=12700)
X = cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values
len(X[0])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
from sklearn.linear_model import LogisticRegression
lgsr = LogisticRegression()
lgsr.fit(X_train,y_train)
y_pre = lgsr.predict(X_test)
y_pred = gnb.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(y_train,lgsr.predict(X_train))
metrics.accuracy_score(y_test,y_pre)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
X_train.shape
for i in range(750):
    plt.scatter(X_train[:,i],y_train)
from sklearn.svm import SVC
svc = SVC(kernel='sigmoid')
svc.fit(X_train,y_train)
y_pr = svc.predict(X_test)
confusion_matrix(y_test,y_pr)
metrics.accuracy_score(y_test,y_pr)
skew = data.skew()
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)

from matplotlib import pyplot
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
pyplot.show()

from pandas.plotting import scatter_matrix
scatter_matrix(data)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train,y_train)
metrics.accuracy_score(y_test,knn.predict(X_test))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',max_depth=4)
dtc.fit(X_train,y_train)
metrics.accuracy_score(y_test,dtc.predict(X_test))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
metrics.accuracy_score(y_test,rfc.predict(X_test))