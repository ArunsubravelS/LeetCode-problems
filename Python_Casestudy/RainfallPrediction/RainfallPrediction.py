import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


pf=pd.read_csv('Rainfall.Csv')
pf.head()#To get first five rows of data
pf.shape #To know number of rows and columns
pf.info() #knowing datatype
pf.describe().T