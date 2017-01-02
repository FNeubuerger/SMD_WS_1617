import numpy as np 
import pandas as pd 

df = pd.read_csv('rapidminer_knn.csv')
df.index = np.arange(len(df))+1461
df.index.name='id'
df.columns= ['SalePrice']
df.to_csv('submission_knn.csv')