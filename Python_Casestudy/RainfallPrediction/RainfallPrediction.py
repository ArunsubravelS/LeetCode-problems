import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
                       
pf=pd.read_csv('Rainfall.Csv')
pf.head()#To get first five rows of data
pf.shape #To know number of rows and columns
pf.info() #knowing datatype
pf.describe()

sb.heatmap(pf.isnull(),yticklabels=False,cbar=False,cmap='viridis')

pf.dropna(how='any', inplace=True)

sb.heatmap(pf.isnull(),yticklabels=False,cbar=False,cmap='viridis')

subdivs = pf['SUBDIVISION'].unique()
num_of_subdivs = subdivs.size
print('Total # of Subdivs: ' + str(num_of_subdivs))
subdivs

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
pf.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'].plot('bar', color='r',width=0.3,title='Subdivision wise Average Annual Rainfall', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Annual Rainfall (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print(pf.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[0,1,2]])
print(pf.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[33,34,35]])

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
pfg = pf.groupby('YEAR').sum()['ANNUAL']
pfg.plot('line', title='Overall Rainfall in Each Year', fontsize=20)

plt.ylabel('Overall Rainfall (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print('Max: ' + str(pfg.max()) + ' ocurred in ' + str(pfg.loc[pfg == pfg.max()].index.values[0:]))
print('Max: ' + str(pfg.min()) + ' ocurred in ' + str(pfg.loc[pfg == pfg.min()].index.values[0:]))
print('Mean: ' + str(pfg.mean()))



months = pf.columns[2:14]
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
xlbls = pf['SUBDIVISION'].unique()
xlbls.sort()
pfg = pf.groupby('SUBDIVISION').mean()[months]
pfg.plot.line(title='Overall Rainfall in Each Month of Year', ax=ax,fontsize=20)
plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
plt.xticks(  rotation = 90)
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper right', fontsize = 'xx-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)

pfg = pfg.mean(axis=0)
print('Max: ' + str(pfg.max()) + ' ocurred in ' + str(pfg.loc[pfg == pfg.max()].index.values[0:]))
print('Max: ' + str(pfg.min()) + ' ocurred in ' + str(pfg.loc[pfg == pfg.min()].index.values[0:]))
print('Mean: ' + str(pfg.mean))

months = pf.columns[2:14]
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
pf.groupby('YEAR').mean()[months].plot.line(title='Overall Rainfall in Each Month of Year', ax=ax,fontsize=20)
#plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
plt.xticks(  rotation = 90)
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper right', fontsize = 'x-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


pf2 = pf[['SUBDIVISION',months[0],months[1],months[2],months[3]]]
#pf2 = pf['YEAR','JAN','FEB','MAR','APR']
pf2.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])

for k in range(1,9):
    pf3 = pf[['SUBDIVISION',months[k],months[k+1],months[k+2],months[k+3]]]
    pf3.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])
    pf2 = pf2.append(pf3)
pf2.index = range(pf2.shape[0])
    
#pf2 = pd.concat([pf2, pd.get_dummies(pf2['SUBDIVISION'])], axis=1)

pf2.drop('SUBDIVISION', axis=1,inplace=True)
#print(pf2.info())
msk = np.random.rand(len(pf2)) < 0.8

pf_train = pf2[msk]
pf_test = pf2[~msk]
pf_train.index = range(pf_train.shape[0])
pf_test.index = range(pf_test.shape[0])

reg = linear_model.LinearRegression()
reg.fit(pf_train.drop('x4',axis=1),pf_train['x4'])
predicted_values = reg.predict(pf_train.drop('x4',axis=1))
residuals = predicted_values-pf_train['x4'].values
print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))
pf_res = pd.DataFrame(residuals)
pf_res.columns = ['Residuals']

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
pf_res.plot.line(title='Different b/w Actual and Predicted (Training Data)', color = 'c', ax=ax,fontsize=20)
ax.xaxis.set_ticklabels([])
plt.ylabel('Residual')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


predicted_values = reg.predict(pf_test.drop('x4',axis=1))
residuals = predicted_values-pf_test['x4'].values
print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))
pf_res = pd.DataFrame(residuals)
pf_res.columns = ['Residuals']

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
pf_res.plot.line(title='Different b/w Actual and Predicted (Test Data)', color='m', ax=ax,fontsize=20)
ax.xaxis.set_ticklabels([])
plt.ylabel('Residual')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


pf2 = pf[['SUBDIVISION',months[0],months[1],months[2],months[3]]]
#pf2 = pf['YEAR','JAN','FEB','MAR','APR']
pf2.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])

for k in range(1,9):
    pf3 = pf[['SUBDIVISION',months[k],months[k+1],months[k+2],months[k+3]]]
    pf3.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])
    pf2 = pf2.append(pf3)
pf2.index = range(pf2.shape[0])
    
#pf2 = pd.concat([pf2, pd.get_dummies(pf2['SUBDIVISION'])], axis=1)

pf2.drop('SUBDIVISION', axis=1,inplace=True)
#print(pf2.info())
msk = np.random.rand(len(pf2)) < 0.8

pf_train = pf2[msk]
pf_test = pf2[~msk]
pf_train.index = range(pf_train.shape[0])
pf_test.index = range(pf_test.shape[0])

reg = linear_model.LinearRegression()
reg.fit(pf_train.drop('x4',axis=1),pf_train['x4'])
predicted_values = reg.predict(pf_train.drop('x4',axis=1))
residuals = predicted_values-pf_train['x4'].values
print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))
pf_res = pd.DataFrame(residuals)
pf_res.columns = ['Residuals']

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
pf_res.plot.line(title='Different b/w Actual and Predicted (Training Data)', color = 'c', ax=ax,fontsize=20)
ax.xaxis.set_ticklabels([])
plt.ylabel('Residual')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


predicted_values = reg.predict(pf_test.drop('x4',axis=1))
residuals = predicted_values-pf_test['x4'].values
print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))
pf_res = pd.DataFrame(residuals)
pf_res.columns = ['Residuals']

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
pf_res.plot.line(title='Different b/w Actual and Predicted (Test Data)', color='m', ax=ax,fontsize=20)
ax.xaxis.set_ticklabels([])
plt.ylabel('Residual')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)



pf2 = pf[['SUBDIVISION',months[0],months[1],months[2],months[3]]]
#pf2 = pf['YEAR','JAN','FEB','MAR','APR']
pf2.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])

for k in range(1,9):
    pf3 = pf[['SUBDIVISION',months[k],months[k+1],months[k+2],months[k+3]]]
    pf3.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])
    pf2 = pf2.append(pf3)
pf2.index = range(pf2.shape[0])
    
#pf2 = pd.concat([pf2, pd.get_dummies(pf2['SUBDIVISION'])], axis=1)

pf2.drop('SUBDIVISION', axis=1,inplace=True)
#print(pf2.info())
msk = np.random.rand(len(pf2)) < 0.8

pf_train = pf2[msk]
pf_test = pf2[~msk]
pf_train.index = range(pf_train.shape[0])
pf_test.index = range(pf_test.shape[0])

reg = linear_model.LinearRegression()
reg.fit(pf_train.drop('x4',axis=1),pf_train['x4'])
predicted_values = reg.predict(pf_train.drop('x4',axis=1))
residuals = predicted_values-pf_train['x4'].values
print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))
pf_res = pd.DataFrame(residuals)
pf_res.columns = ['Residuals']

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
pf_res.plot.line(title='Different b/w Actual and Predicted (Training Data)', color = 'c', ax=ax,fontsize=20)
ax.xaxis.set_ticklabels([])
plt.ylabel('Residual')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


predicted_values = reg.predict(pf_test.drop('x4',axis=1))
residuals = predicted_values-pf_test['x4'].values
print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))
pf_res = pd.DataFrame(residuals)
pf_res.columns = ['Residuals']

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
pf_res.plot.line(title='Different b/w Actual and Predicted (Test Data)', color='m', ax=ax,fontsize=20)
ax.xaxis.set_ticklabels([])
plt.ylabel('Residual')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)



pf.hist(figsize=(18,18))