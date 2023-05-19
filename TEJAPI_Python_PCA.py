#!/usr/bin/env python
# coding: utf-8

# In[164]:


#找出0050現存成分股
import tejapi
import pandas as pd

tejapi.ApiConfig.api_key = "Your Key"
tejapi.ApiConfig.ignoretz = True

mdate = {'gte':'2000-01-01', 'lte':'2022-11-24'}
data = tejapi.get('TWN/EWISAMPLE',
                  idx_id = "IX0002",
                  start_date = mdate,
                          paginate=True)

data1 = data[data["end_date"] < "2022-11-24"]
diff_data = pd.concat([data,data1,data1]).drop_duplicates(keep=False)
coid = list(diff_data["coid"])
print(len(coid))
diff_data


# In[166]:


#下載期間報酬率並合併
for i in range(0,len(coid)):
    print(i)
    if i == 0:
        df = tejapi.get('TWN/EWPRCD2',
                                  coid = coid[i],
                                  mdate = {'gte':'2013-01-01', 'lte':'2022-11-24'},
                                  paginate=True)
        df.set_index(df["mdate"],inplace=True)
        Df = pd.DataFrame({coid[i]:df["roia"]})
    else:
        df = tejapi.get('TWN/EWPRCD2',
                                  coid = coid[i],
                                  mdate = {'gte':'2013-01-01', 'lte':'2022-11-24'},
                                  paginate=True)
        df.set_index(df["mdate"],inplace=True)
        Df1 = pd.DataFrame({coid[i]:df["roia"]})
        Df = pd.merge(Df,Df1[coid[i]],how='left', left_index=True, right_index=True)


# In[169]:


#剔除3711、5876、6415
del Df["3711"]
del Df["5876"]
del Df["6415"]
Df


# In[170]:


#畫圖套件
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[172]:


#資料視覺化
cor = Df.corr()
plt.figure(figsize=(30,30))
plt.title("Correlation Matrix")
sns.heatmap(cor, vmax=1,square=True,annot=True,cmap="cubehelix")


# In[174]:


#標準化
scale = StandardScaler().fit(Df)
rescale = pd.DataFrame(scale.fit_transform(Df),columns=Df.columns,index=Df.index)
#標準化視覺化
plt.figure(figsize=(20,5))
plt.title("2330_Return")
rescale["2330"].plot()
plt.grid=True
plt.legend()
plt.show()


# In[300]:


X_train = rescale.copy()


# In[301]:


#主成分分析
#套件
from sklearn.decomposition import PCA
#主成分個數
n_components = 10
pca = PCA(n_components=n_components)
Pc = pca.fit(X_train)


# In[305]:


#PCA解釋變數
fig, axes = plt.subplots(ncols=2)
Series1 = pd.Series(Pc.explained_variance_ratio_[:n_components ]).sort_values()
Series2 = pd.Series(Pc.explained_variance_ratio_[:n_components ]).cumsum()

Series1.plot.barh(title="Explained Variance",ax=axes[0])
Series2.plot(ylim=(0,1),ax=axes[1],title="Cumulative Explained Variance")
print("變數累積解釋比例：")
print(Series2[len(Series2)-1:len(Series2)].values[0])
print("各變數解釋比例：")
print(Series1.sort_values(ascending=False))


# In[306]:


#檢視投組權重
n_components = 10
weights = pd.DataFrame()
for i in range(n_components):
    weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])
weights = weights.values.T
weight_port = pd.DataFrame(weights,columns=Df.columns)
weight_port.index = [f'Portfolio{i}' for i in range(weight_port.shape[0])]
weight_port


# In[308]:


pca.components_[0]


# In[232]:


#權重合為1
weight_port.sum(axis = 1)


# In[234]:


#畫權重圖top5
weight_port[:5].T.plot.bar(subplots=True,layout = (int(5),1),figsize=(20,25),
                      legend=False,sharey=True,ylim=(-2,2))


# In[309]:


weight_port.iloc[0].T.sort_values(ascending=False).plot.bar(subplots=True,figsize=(20,5),
                          legend=False,sharey=True,ylim=(-0.75,0.75))


# In[310]:


#尋找最佳特徵投組(由Sharpe Ratio決定)
def sharpe_ratio(ts_returns):
    ts_returns = ts_returns
    days = ts_returns.shape[0]
    n_years = days/252
    if ts_returns.cumsum()[-1] < 0:  
        annualized_return = (np.power(1+abs(ts_returns.cumsum()[-1])*0.01,1/n_years)-1)*(-1)
    else:
        annualized_return = np.power(1+abs(ts_returns.cumsum()[-1])*0.01,1/n_years)-1
    annualized_vol = (ts_returns*0.01).std()*np.sqrt(252)
    annualized_sharpe = annualized_return / annualized_vol
    
    return annualized_return,annualized_vol,annualized_sharpe


# In[311]:


n_components = 10
annualized_ret = np.array([0.]*n_components)
sharpe_metric = np.array([0.]*n_components)
annualized_vol = np.array([0.]*n_components)
coids = X_train.columns.values
n_coids = len(coids)


# In[312]:


#主成分分析
pca = PCA(n_components=n_components)
Pc = pca.fit(X_train)
pcs = pca.components_
for i in range(n_components):
    pc_w = pcs[i] / sum(pcs[i])
    eigen_port = pd.DataFrame(data={"weights":pc_w.squeeze()},index=coids)
    eigen_port.sort_values(by=["weights"],ascending=False,inplace=True)
    #權重與每天報酬內積，得出每日投資組合報酬
    eigen_port_returns = np.dot(X_train.loc[:,eigen_port.index],eigen_port["weights"])
    eigen_port_returns = pd.Series(eigen_port_returns.squeeze(),
                                   index = X_train.index)
    
    ar,vol,sharpe = sharpe_ratio(eigen_port_returns)
    
    annualized_ret[i] = ar
    annualized_vol[i] = vol
    sharpe_metric[i] = sharpe

sharpe_metric = np.nan_to_num(sharpe_metric)


# In[313]:


#Result of top 5
N=5
result = pd.DataFrame({"Annual Return":annualized_ret,"Vol":annualized_vol,"Sharpe":sharpe_metric})
result.dropna(inplace=True)
#Sharpe Ratio of PCA portfolio
ax = result[:N]["Sharpe"].plot(linewidth=3,xticks=range(0,N,1))
ax.set_ylabel("Sharpe")
result.sort_values(by=["Sharpe"],ascending=False,inplace=True)
print(result[:N])


# In[155]:


X_train


# In[156]:


#0050
import tejapi
import pandas as pd

tejapi.ApiConfig.api_key = "Your Key"
tejapi.ApiConfig.ignoretz = True
market = tejapi.get('TWN/EWPRCD2',
                          coid = "0050",
                          mdate = {'gte':'2013-12-12', 'lte':'2020-08-31'},
                          paginate=True)
market.set_index(market["mdate"],inplace=True)
market = pd.DataFrame({"0050":market["roia"]})

m_ar,m_vol,m_sharpe = sharpe_ratio(market["0050"])
print(m_ar,m_vol,m_sharpe)
market["0050"].cumsum().plot(rot=45)


# In[323]:


#回測
def Backtest(i,data):
    pca = PCA()
    Pc = pca.fit(data)
    pcs = pca.components_
    pc_w = pcs[i] / sum(pcs[i])
    eigen_port = pd.DataFrame(data={"weights":pc_w.squeeze()},index=coids)
    eigen_port.sort_values(by=["weights"],ascending=False,inplace=True)
    #權重與每天報酬取內積得出每日投資組合報酬
    eigen_port_returns = np.dot(data.loc[:,eigen_port.index],eigen_port["weights"])
    eigen_port_returns = pd.Series(eigen_port_returns.squeeze(),
                                   index = data.index)
    
    ar,vol,sharpe = sharpe_ratio(eigen_port_returns)
    return eigen_port_returns,ar,vol,sharpe


# In[322]:


#畫出投資組合權重
def Weight_plot(i):
    top_port = weight_port.iloc[[i]].T
    port_name = top_port.columns.values.tolist()
    top_port.sort_values(by=port_name,ascending=False,inplace=True)
    ax = top_port.plot(title = port_name[0],xticks=range(0,len(coids),1),
                  figsize=(15,6),
                  rot=45,linewidth=3)
    ax.set_ylabel("Portfolio Weight")


# In[324]:


#Backtest
portfolio = 0
train_returns,train_ar,train_vol,train_sharpe = Backtest(portfolio,X_train)
ax = train_returns.cumsum().plot(rot=45)
ax.set_ylabel("Accumulated Return(%)")
Weight_plot(portfolio)


# In[318]:


#Test_Backtest
portfolio = 2
test_returns,test_ar,test_vol,test_sharpe = Backtest(portfolio,X_test)
print(test_ar,test_vol,test_sharpe)
ax = test_returns.cumsum().plot(rot=45)
ax.set_ylabel("Accumulated Return(%)")
Weight_plot(portfolio)

