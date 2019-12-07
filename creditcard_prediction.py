import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv('train.csv')

df=df.drop('Id',axis=1)
x=df.drop('Category',axis=1)
y=df.Category
#转换成数值型数据
d = defaultdict(LabelEncoder)
x_trains = x.apply(lambda x: d[x.name].fit_transform(x))
# print(x_trans.shape)
x_train, x_test, y_train, y_test = train_test_split(x_trains, y, test_size=0.2)
# 决策树不纯度
# from sklearn.model_selection import GridSearchCV
# thresholds = np.linspace(0, 0.2, 50)
# param_grid = {'min_impurity_decrease':thresholds}
# clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=5)
# clf.fit(x,y)
# print("best_parms:{0}\nbest_score:{1}".format(clf.best_params_, clf.best_score_))
#
# test=pd.read_csv('test.csv')
# Id=test.Id
# # print(df.shape)
# test=test.drop('Id',axis=1)
# print(test.shape)
# outcome=clf.predict(test)
# np.savetxt('new.csv', outcome, fmt='%d', delimiter=',')
# print(clf.predict(test))
# 决策树深度调优
# def cv_score(d):
#     clf = tree.DecisionTreeClassifier(max_depth=d)
#     clf=clf.fit(x_train, y_train)
#     return(clf.score(x_train, y_train), clf.score(x_test, y_test))
# deepth=np.arange(1,10)
# score=[cv_score(d) for d in deepth]
# tr_score=[s[0] for s in score]
# te_score=[s[1] for s in score]
# tr_best=np.argmax(tr_score)
# te_best=np.argmax(te_score)
# print("bestdepth:", te_best+1, " bestdepth_score:", te_score[te_best], '\n')
# clf = tree.DecisionTreeClassifier(max_depth=1)
# clf = clf.fit(x_train, y_train)
# print("train score:", clf.score(x_train, y_train))
# print("test score:", clf.score(x_test, y_test))
test=pd.read_csv('test.csv')
Id=test.Id
test=test.drop('Id',axis=1)
# print(test.shape)
# outcome=clf.predict(test)
# print(outcome)
# np.savetxt('new.csv', outcome, fmt='%d', delimiter=',')
# 决策树可视化
# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.view()


def fit_model(alg,parameters):
    # X=x_train
    # y=y_train  #由于数据较少，使用全部数据进行网格搜索
    scorer=make_scorer(roc_auc_score)  #使用roc_auc_score作为评分标准
    grid = GridSearchCV(alg,parameters,scoring=scorer,cv=5)  #使用网格搜索，出入参数
    start=time()  #计时
    grid=grid.fit(x,y)  #模型训练
    end=time()
    t=round(end-start,3)
    print (grid.best_params_)  #输出最佳参数
    print ('searching time for {} is {} s'.format(alg.__class__.__name__,t)) #输出搜索时间
    print("train score:", grid.score(x, y))
    # print("test score:", grid.score(x_test, y_test))
    return grid

#列出需要使用的算法
alg1=DecisionTreeClassifier(random_state=29)
alg2=SVC(probability=True,random_state=29)  #由于使用roc_auc_score作为评分标准，需将SVC中的probability参数设置为True
alg3=RandomForestClassifier(random_state=29)
alg4=AdaBoostClassifier(random_state=29)
alg5=KNeighborsClassifier(n_jobs=-1)

#列出需要调整的参数范围
parameters1={'max_depth':range(1,10),'min_samples_split':range(2,10)}
parameters2 = {"C":range(1,20), "gamma": [0.05,0.1,0.15,0.2,0.25]}
parameters3_1 = {'n_estimators':range(10,200,10)}
parameters3_2 = {'max_depth':range(1,10),'min_samples_split':range(2,10)}  #搜索空间太大，分两次调整参数
parameters4 = {'n_estimators':range(10,200,10),'learning_rate':[i/10.0 for i in range(5,15)]}
parameters5 = {'n_neighbors':range(2,10),'leaf_size':range(10,80,20) }

clf1=fit_model(alg1,parameters1)
clf2=fit_model(alg2,parameters2)
clf3_m1=fit_model(alg3,parameters3_1)
alg3=RandomForestClassifier(random_state=29,n_estimators=180)
clf3=fit_model(alg3,parameters3_2)
clf4=fit_model(alg4,parameters4)
clf5=fit_model(alg5,parameters5)
def save(clf,i):
    category=clf.predict(test)
    sub=pd.DataFrame({ 'Id': Id, 'Category': category })
    sub.to_csv("res_tan_{}.csv".format(i), index=False)

i=1
for clf in [clf1,clf2,clf3,clf4,clf5]:
    save(clf,i)
    i=i+1