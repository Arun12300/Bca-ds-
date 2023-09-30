import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('insurance (1).csv')

data.columns
data.info()
data.describe()
#########################
sns.heatmap(data.corr(),annot=True)
sns.heatmap(data.corr())

#####age verse charge
##scatter polt is used to one is numercal data second is another numerical

sns.scatterplot(x=data['age'],y=data['charges'])
sns.scatterplot(x=data['bmi'],y=data['charges'])

#gender verse charges
sns.boxplot(x=data['sex'],y=data['charges'])
sns.barplot(x=data['sex'],y=data['charges'])

########children  verse charges
####box polt is used to one is numercal data second ic char data
sns.boxplot(x=data['children'],y=data['charges'])

########### smoker verse charges  
sns.boxplot(x=data['smoker'],y=data['charges'])
#region verse
sns.boxplot(x=data['region'],y=data['charges'])


###############################
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder() 
columns=['sex','smoker', 'region']
for column in columns:
    data[column]=encoder.fit_transform(data[column])
###############################
x=data.drop(['charges'],axis=1)
y=data['charges']
##############################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.25,random_state=0)
###############################

#create a model
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
sel=SelectFromModel(Lasso(alpha=0.05))
sel.fit(x_train,y_train)
sel.get_support()
x.columns[sel.get_support()]

x_train=x_train.loc[:,['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
x_test=x_test.loc[:,['age', 'sex', 'bmi', 'children', 'smoker', 'region']]

from sklearn.linear_model import Lasso
regressor1=Lasso(alpha=0.05)
regressor1.fit(x_train,y_train)
####################################
regressor1.coef_
regressor1.intercept_

###########################
y_pred1=regressor1.predict(x_test)

######################################
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred1))
metrics.mean_absolute_error(y_test,y_pred1)
metrics.r2_score(y_test,y_pred1)
##########################################################
################### Ridge

from sklearn.linear_model import Ridge
regressor2=Ridge(alpha=0.09)
regressor2.fit(x_train,y_train)
####################################3
regressor2.coef_
regressor2.intercept_
y_pred2=regressor2.predict(x_test)
#####################################
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred2))
metrics.mean_absolute_error(y_test,y_pred2)
metrics.r2_score(y_test,y_pred2)





from sklearn.linear_model import ElasticNet
regressor3=ElasticNet(alpha=0.09)
regressor3.fit(x_train,y_train)
####################################3
regressor3.coef_
regressor3.intercept_
y_pred3=regressor3.predict(x_test)


from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred3))
metrics.mean_absolute_error(y_test,y_pred3)
metrics.r2_score(y_test,y_pred3)

####################################
 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_
###############################
y_pred=regressor.predict(x_test)
###############################
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
metrics.mean_absolute_error(y_test, y_pred)
metrics.r2_score(y_test, y_pred)
###############################
plt.scatter(y_pred,y_test,color='Blue')





























