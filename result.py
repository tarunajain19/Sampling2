import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
# from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
df=pd.read_csv('Creditcard_data.csv')
# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)


# define undersampling strategy
undersample = RandomUnderSampler(sampling_strategy='majority')

# fit and apply the transform
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)


#PART 2
# import SVM libraries 
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

model=SVC()
clf_under = model.fit(X_train_under, y_train_under)
pred_under = clf_under.predict(X_test)

print("ROC AUC score for undersampled data using SVM: ", roc_auc_score(y_test, pred_under))

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train_under, y_train_under)
pred_under=clf.predict(X_test)
print("ROC AUC score for undersampled data using random forest: ", roc_auc_score(y_test, pred_under))

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train_under, y_train_under)
pred_under=model.predict(X_test)
print("ROC AUC score for undersampled data using Naive byes: ", roc_auc_score(y_test, pred_under))

clf1 = DecisionTreeClassifier()
clf1.fit(X_train_under, y_train_under)
pred_under=clf1.predict(X_test)
print("ROC AUC score for undersampled data using Decision Tree Classifier: ", roc_auc_score(y_test, pred_under))


#PART 1
# import SMOTE oversampling and other necessary libraries 
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#import data

df = pd.read_csv('Creditcard_data.csv')

# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)


# define oversampling strategy
SMOTE = SMOTE()

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)


#PART 2
# import SVM libraries 
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

model=SVC()
clf_SMOTE = model.fit(X_train_SMOTE, y_train_SMOTE)
pred_SMOTE = clf_SMOTE.predict(X_test)

print("ROC AUC score for oversampled SMOTE data: ", roc_auc_score(y_test, pred_SMOTE))


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train_SMOTE, y_train_SMOTE)
pred_SMOTE=clf.predict(X_test)
print("ROC AUC score for oversampled data using random forest: ", roc_auc_score(y_test, pred_SMOTE))

from sklearn.naive_bayes import GaussianNB
model2 = GaussianNB()
model2.fit(X_train_SMOTE, y_train_SMOTE)
pred_SMOTE=model2.predict(X_test)
print("ROC AUC score for oversampled data using Naive bayes: ", roc_auc_score(y_test, pred_SMOTE))


#PART 1
# import sampling and other necessary libraries 
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

#import data

df = pd.read_csv('Creditcard_data.csv')

# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# define pipeline
model = SVC()
over = SMOTE(sampling_strategy=0.4)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under), ('model', model)]
pipeline = Pipeline(steps=steps)
#PART 2
# import libraries for evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from numpy import mean

# evaluate pipeline
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=5, n_jobs=-1)
score = mean(scores)
print('ROC AUC score for the combined sampling method using SVM: %.3f' % score)


clf = RandomForestClassifier(max_depth=2, random_state=0)
over=SMOTE(sampling_strategy=0.4)
under=RandomUnderSampler(sampling_strategy=0.5)
steps=[('o', over), ('u', under), ('clf', clf)]
pipeline = Pipeline(steps=steps)
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=5, n_jobs=-1)
score = mean(scores)
print("ROC AUC score for the combined sampling using random forest: %.3f " % score )


from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
over=SMOTE(sampling_strategy=0.4)
under=RandomUnderSampler(sampling_strategy=0.5)
steps=[('o', over), ('u', under), ('model3', model3)]
pipeline = Pipeline(steps=steps)
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=5, n_jobs=-1)
score = mean(scores)
print("ROC AUC score for the combined sampling using naive bayes: %.3f " % score )

