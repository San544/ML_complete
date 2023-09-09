import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
data=pd.read_csv('data1.csv')

def preprocess_dataset(data):
  
  #handling missing values 

  #drop rows with missing values
  data.dropna()
  
  #replace missing values with 0
  data.fillna(0)

  #replace missing values with column mean
  data.fillna(data.mean())
  #replace missing values with column median
  data.fillna(data.median())
  #replace missing values with the mode of the column
  data.fillna(data.mode().iloc[0])
  
  
  
  
  #one hot encoding
  encoded_data=pd.get_dummies(data)
  print(encoded_data.head())

  #standardization 
  from sklearn.preprocessing import StandardScaler
  scaler=StandardScaler()
  scaled_data=scaler.fit_transform(data)


  #Minmax Scaling:

  from sklearn.preprocessing import MinMaxScaler
  scaler1=MinMaxScaler()
  min_max_scaled_data=scaler1.fit_transform(data)

  #splitting dataset into test and train
  from sklearn.model_selection import train_test_split
  X = data.iloc[:,:-1]
  y = data.iloc[:,-1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  #oversampling Minority class
  from imblearn.over_sampling import SMOTE
  oversampler=SMOTE()
  X_resampled,y_resampled= oversampler.fit_resample(X_train,y_train)
  
  #undersampling majority class
  from imblearn.under_sampling import RandomUnderSampler
  undersampler=RandomUnderSampler()
  X_resampled1,y_resampled1=undersampler.fit_resample(X_train,y_train)

  #Removing features with low variance
  from sklearn.feature_selection import VarianceThreshold
  selector=VarianceThreshold(data,threshold=0.1)
  selected_features=selector.fit_transform(data)
  
  
  #selecting k best features
  from sklearn.feature_selection import SelectKBest,f_classif
  selectk=SelectKBest(score_func=f_classif,k=5)
  kbest_features=selectk.fit_transform(X_train,y_train)
  print(data.head())
preprocess_dataset(data)
