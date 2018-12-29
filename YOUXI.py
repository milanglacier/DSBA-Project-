
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import time
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")


# # 分类问题 数据预处理

# In[2]:


dataset = pd.read_csv('tap_fun_train.csv')
dataset.head()


# In[3]:


dataset['ifpay'] = 0
dataset['willpay'] = 0
dataset.head()


# In[4]:


dataset.loc[dataset.pay_price>0,'ifpay'] = 1
dataset.loc[dataset.prediction_pay_price>0,'willpay'] = 1
#玩家如果付费,就记为1,否则为0,用于后面的分类任务


# In[5]:


print(dataset['ifpay'].value_counts())
print(dataset['willpay'].value_counts())


# In[42]:


df_clf = dataset.drop(["prediction_pay_price",'pay_count'],axis=1)
#将付费金额数值列删除,先做分类问题


# In[43]:


df_clf['register_time'][0]


# In[44]:


for index, component in enumerate(['year', 'month', 'day']):
    df_clf['%s_%s'%('register_time', component)] = df_clf['register_time'].        apply(lambda x: int(x.split(' ')[0].split('-')[index]))
#将注册时间分拆为年-月-日


# In[45]:


df_clf = df_clf.drop(['register_time'],axis = 1)
df_clf.head()
#将注册时间(字符串类型)那一列删除


# In[46]:


df_clf['register_days'] = df_clf['register_time_month']*31+df_clf['register_time_day']


# In[47]:


df_clf.get_dtype_counts()


# In[48]:


x = df_clf.drop(['user_id','ifpay','willpay'],axis=1)
y = df_clf['willpay']


# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


x_train, x_test, y_train, y_test = train_test_split(
                                     x, y, test_size=0.3, stratify = y)
#这是一个严重样本不平衡问题,所以必须按照是否付费的比例进行训练-测试集分割


# In[51]:


import matplotlib.pyplot as plt
from sklearn import metrics


# In[52]:


def model_performance(model_name, clf, x_train, y_train, y_test, y_pred):

    print('Model name: %s'%model_name )
    print('Test precision rate: %f'%metrics.precision_score(y_test, y_pred) )
    print('Test ROC AUC Score: %f'%metrics.roc_auc_score(y_test, y_pred) )
    print('Test recall rate: %f'%metrics.recall_score(y_test,y_pred))

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred) 
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic') 
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc) 
    plt.legend(loc='lower right') 
    plt.plot([0,1],[0,1],'r--') 
    plt.xlim([-0.1,1]) 
    plt.ylim([-0.1,1])
    plt.ylabel('True Positive Rate') 
    plt.xlabel('False Positive Rate') 
    plt.show()


# # 决策树部分

# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)


# In[19]:


y_pred = tree.predict(x_test)


# In[20]:


model_performance('decision tree',tree, x_train,y_train,y_test,y_pred)


# # 欠采样决策树

# In[53]:


from imblearn.under_sampling import RandomUnderSampler


# In[54]:


rus = RandomUnderSampler(ratio=0.7,random_state=134)
x_udtrain, y_udtrain = rus.fit_sample(x_train, y_train)


# # 欠采样随机森林

# In[69]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[57]:


rf = RandomForestClassifier(n_estimators=50,n_jobs=-1)


# In[58]:


rf.fit(x_udtrain,y_udtrain)


# In[59]:


y_rf = rf.predict(x_test)


# In[60]:


model_performance('random forests undersampled', rf, x_udtrain,y_udtrain,y_test,y_rf)


# In[97]:


def CV(model,x,y,criterion):
    start = time.clock()
    score = cross_val_score(model,x,y,cv=5,scoring=criterion)
    plt.plot(score)
    plt.title('%s rate of five folds CV %f '%(criterion,np.mean(score)))
    plt.show()
    end = (time.clock()-start)
    print('运行时间为 %.1f 秒'%end)


# In[76]:


CV(rf,x_udtrain,y_udtrain,'recall')


# # 分类问题二:判断玩家7-45天是否会付费
#    数据预处理

# In[5]:


df_clf2 = dataset.drop(['user_id','register_time','ifpay','willpay'],axis=1)
df_clf2 = df_clf2[df_clf2.prediction_pay_price>0]


# In[6]:


for index, component in enumerate(['year', 'month', 'day']):
    df_clf2['%s_%s'%('register_time', component)] = dataset['register_time'].        apply(lambda x: int(x.split(' ')[0].split('-')[index]))
#将注册时间分拆为年-月-日
df_clf2['register_days'] = df_clf2['register_time_month']*31+df_clf2['register_time_day']
df_clf2 = df_clf2.drop(['register_time_year'],axis=1)


# In[7]:


df_clf2['continue_paying'] = df_clf2['prediction_pay_price']-df_clf2['pay_price']
df_clf2.continue_paying = df_clf2.continue_paying.apply(lambda x:int(bool(x)))
df_clf2.head()


# In[8]:


u = df_clf2.drop(['prediction_pay_price','continue_paying'],axis=1)
z = df_clf2['continue_paying']


# In[11]:


u_train, u_test, z_train, z_test = train_test_split(
                                     u, z, test_size=0.3,stratify=z)


# # 回归 数据预处理

# In[13]:


def regressor_performance(model_name,z_test,z_pred):   
    print('the R2 of %s is %f'%(model_name,metrics.r2_score(z_test,z_pred)))
    plt.plot(range(np.shape(z_test)[0]),z_test,color='#FF5685',label='true value')
    plt.plot(range(np.shape(z_test)[0]),z_pred,color='#6AC7E6',label='predict value')
    plt.legend(loc='upper right')
    plt.show()


# In[78]:


df_rg = dataset.drop(['ifpay','willpay','register_time'],axis=1)
df_rg = df_rg[df_rg['prediction_pay_price']>0]


# In[79]:


for index, component in enumerate(['year', 'month', 'day']):
    df_rg['%s_%s'%('register_time', component)] = dataset['register_time'].        apply(lambda x: int(x.split(' ')[0].split('-')[index]))
#将注册时间分拆为年-月-日
df_rg['register_days'] = df_rg['register_time_month']*31+df_rg['register_time_day']


# In[80]:


u = df_rg.drop(['prediction_pay_price','user_id'],axis=1)
z = df_rg['prediction_pay_price']


# In[81]:


u_train, u_test, z_train, z_test = train_test_split(
                                     u, z, test_size=0.3)


# In[82]:


def regressor_performance(model_name,z_test,z_pred):   
    print('the R2 of %s is %f'%(model_name,metrics.r2_score(z_test,z_pred)))
    plt.plot(range(np.shape(z_test)[0]),z_test,color='#FF5685',label='true value')
    plt.plot(range(np.shape(z_test)[0]),z_pred,color='#6AC7E6',label='predict value')
    plt.legend(loc='upper right')
    plt.show()


# # Lasso 回归

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


lasso = Lasso()


# In[ ]:


lasso.fit(u_train,z_train)
z_ls = lasso.predict(u_test)


# In[ ]:


regressor_performance('lasso',z_test,z_ls)
CV(lasso,u,z,None)


# # 多分类问题

# In[38]:


def predictClass(paymoney):
    global ability
    if paymoney <= 6:
        ability = 1
    elif paymoney <= 68:
        ability = 2
    elif paymoney <= 328:
        ability = 3
    elif paymoney <= 2000:
        ability = 4
    else:
        ability = 5
    return(ability)


# In[39]:


df_clf3 = df_clf2


# In[41]:


df_clf3['predict_class'] = df_clf3['prediction_pay_price'].apply(predictClass)


# In[43]:


h = df_clf3.drop(['prediction_pay_price','predict_class'],axis=1)
g = df_clf3.predict_class


# In[44]:


h_train, h_test, g_train, g_test = train_test_split(
                                     h, g, test_size=0.3, stratify = g)


# # 随机森林多分类器

# In[45]:


rf3 = RandomForestClassifier(n_estimators=200,n_jobs=-1)


# In[46]:


start = time.clock()
rf3.fit(h_train,g_train)
end = (time.clock()-start)
print('运行时间为 %.1f 秒'%end)


# In[47]:


g_rf3 = rf3.predict(h_test)
g_trrf3 = rf3.predict(h_train)
rf3.score(h_test,g_test)


# In[53]:


metrics.f1_score(g_test,g_rf3,average='macro')


# In[56]:


CV(rf3,h,g,'accuracy')


# # 特征工程及将数据导出至R

# In[297]:


dataR2 = dataset


# In[298]:


for index, component in enumerate(['year', 'month', 'day']):
    dataR2['%s_%s'%('register_time', component)] = dataset['register_time'].        apply(lambda x: int(x.split(' ')[0].split('-')[index]))
dataR2['register_days'] = dataR2['register_time_month']*31+dataR2['register_time_day']
dataR2 = dataR2.drop(['register_time_year'],axis=1)
#该列表示玩家注册的日子距离2018/1/1多久
#将注册时间分拆为年-月-日


# In[299]:


dataR2['basic_material_add'] = dataR2.wood_add_value+dataR2.stone_add_value+                               dataR2.ivory_add_value+dataR2.meat_add_value+                              dataR2.magic_add_value


dataR2['basic_material_reduce'] = dataR2.wood_reduce_value+dataR2.stone_reduce_value +                                 dataR2.ivory_reduce_value+dataR2.meat_reduce_value +                                 dataR2.magic_reduce_value


# In[301]:


dataR2['proUnit_add'] = dataR2.infantry_add_value+dataR2.cavalry_add_value+                        dataR2.shaman_add_value+dataR2.wound_cavalry_add_value+                        dataR2.wound_infantry_add_value+dataR2.wound_shaman_add_value
dataR2['proUnit_reduce'] = dataR2.infantry_reduce_value+dataR2.cavalry_reduce_value+                            dataR2.shaman_reduce_value+dataR2.woubnd_cavalry_reduce_value+                            dataR2.wound_infantry_reduce_value+dataR2.shaman_reduce_value


# In[302]:


dataR2['acceleration_add'] = dataR2.general_acceleration_add_value+dataR2.building_acceleration_add_value+                            dataR2.reaserch_acceleration_add_value+dataR2.training_acceleration_add_value+                            dataR2.treatment_acceleraion_add_value
dataR2['acceleration_reduce'] = dataR2.general_acceleration_reduce_value+dataR2.building_acceleration_reduce_value+                               dataR2.reaserch_acceleration_reduce_value+dataR2.training_acceleration_reduce_value+                               dataR2.training_acceleration_reduce_value


# In[303]:


dataR2['building_level'] = dataR2.bd_barrack_level+dataR2.bd_dolmen_level+                            dataR2.bd_guest_cavern_level+dataR2.bd_hall_of_war_level+                            dataR2.bd_healing_lodge_level+dataR2.bd_healing_spring_level+                            dataR2.bd_hero_gacha_level+dataR2.bd_hero_pve_level+                            dataR2.bd_hero_strengthen_level+dataR2.bd_magic_coin_tree_level


# In[310]:


dataR2['research_level'] = 0
for i in dataR2.columns[50:99]:
    dataR['research_level'] += dataR2[i]


# In[313]:


dataR2['basic_material_ratio'] = dataR2.basic_material_add/(dataR2.basic_material_reduce+1)


# In[314]:


dataR2['proUnit_ratio'] = dataR2.proUnit_add/(dataR2.proUnit_reduce+1)


# In[316]:


dataR2['active_pvp_winrate'] = dataR2.pvp_win_count/(dataR2.pvp_lanch_count+0.5)
dataR2['pvp_win_rate'] = dataR2.pvp_win_count/(dataR2.pvp_battle_count+0.5)
dataR2.loc[dataR2.active_pvp_winrate<=0,'active_pvp_winrate'] = 0
dataR2.loc[dataR2.pvp_win_rate<=0,'pvp_win_rate'] = 0


# In[319]:


dataR2['active_pve_winrate'] = dataR2.pve_win_count/(dataR2.pve_lanch_count+0.5)
dataR2['pve_win_rate'] = dataR2.pve_win_count/(dataR2.pve_battle_count+0.5)
dataR2.loc[dataR2.active_pve_winrate<=0,'active_pve_winrate'] = 0
dataR2.loc[dataR2.pve_win_rate<=0,'pve_win_rate'] = 0


# In[342]:


dataR2 = dataR2.iloc[:,99:128]


# In[406]:


dataR2 = dataR2[dataR2.prediction_pay_price>0]
dataR2['pay_class'] = dataR1['prediction_pay_price'].apply(predictClass)
dataR2.to_csv('dataR2.csv')

