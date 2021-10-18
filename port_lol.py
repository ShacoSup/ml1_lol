#%% 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터 불러온 후 분할
path = r'C:\Users\USER\Videos\port\data_pre.csv'
raw_data = pd.read_csv(path)
X = raw_data.drop(['Win', 'Team', 'Player', 'Role'], axis=1)
y = raw_data.Win
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.25, random_state = 0)
print("train X Data shape: {}".format(X_train.shape))
print("val X Data shape: {}".format(X_val.shape))
print("test X Data shape: {}".format(X_test.shape))

X_train.head
# =============================================================================
# 데이터 불러온 후 분할, 히트맵으로  확인, 산점도로  확인, 차원축소를 위한 주성분 분석
# 각각의 스케일링, 차원축소+스케일러+아무것도 안했을때 각각 조합을 비교
# 판별분석, 로지스틱 회귀분석, KNN, TREE_DECISION, SVM
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

# # 히트맵
# f, ax = plt.subplots(figsize=(15, 15))
# sns.heatmap(raw_data.corr(), annot = True, linewidths = .5, fmt = '.2f', ax = ax)
# plt.show()

# # 산점도
# sns.pairplot(raw_data, hue = "Win", height = 2.5)
# plt.tight_layout()
# plt.show()

# 특성변수의 분포 살펴보기
X_train.boxplot(figsize=(10, 8), vert = False)

col = X_train.columns.values
# 각각 스케일링
# StandardScaler 
# 평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 
# 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
from sklearn.preprocessing import StandardScaler  #z값을 사용하는 scaler
SDsc = StandardScaler()
X_train_SDsc = SDsc.fit_transform(X_train)
X_val_SDsc = SDsc.transform(X_val)
X_test_SDsc = SDsc.transform(X_test)
x1 = pd.DataFrame(X_train_SDsc)
x1.columns = [col]
# pd.DataFrame(x1).boxplot(figsize=(10,7), vert = False)

# RobustScaler
# 아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 
# StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.
from sklearn.preprocessing import RobustScaler
RBsc = RobustScaler()
X_train_RBsc = RBsc.fit_transform(X_train)
X_val_RBsc = RBsc.transform(X_val)
X_test_RBsc = RBsc.transform(X_test)
x2 = pd.DataFrame(X_train_RBsc)
# x2.columns = [col]
# pd.DataFrame(x2).boxplot(figsize=(10,7), vert = False)

# MinmaxScaler
# 모든 feature 값이 0~1사이에 있도록 데이터를 재조정한다. 데이터에 음수값이 없을때 사용
# 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다.
# from sklearn.preprocessing import MinMaxScaler
# MMsc = MinMaxScaler()
# X_train_MMsc = MMsc.fit_transform(X_train)
# X_val_MMsc = MMsc.transform(X_val)
# X_test_MMsc = MMsc.transform(X_test)
# x3 = pd.DataFrame(X_train_MMsc)
# x3.columns = [col]
# pd.DataFrame(x3).boxplot(figsize=(10,7), vert = False)

# MaxAbsScaler
# 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 
# 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.
# from sklearn.preprocessing import MaxAbsScaler
# MAsc = MaxAbsScaler()
# X_train_MAsc = MAsc.fit_transform(X_train)
# X_val_MAsc = MAsc.transform(X_val)
# X_test_MAsc = MAsc.transform(X_test)
# x4 = pd.DataFrame(X_train_MAsc)
# x4.columns = [col]
# pd.DataFrame(x4).boxplot(figsize=(10,7), vert = False)

#%% PCA
# SD
from sklearn.decomposition import PCA
pca_SD = PCA(n_components = 3).fit(X_train_SDsc)
X_train_pca_SD = pca_SD.transform(X_train_SDsc)
X_val_pca_SD = pca_SD.transform(X_val_SDsc)
X_test_pca_SD = pca_SD.transform(X_test_SDsc)
print("Original shape: {}".format(str(X_train_SDsc.shape)))
print("Reduced shape: {}".format(str(X_train_pca_SD.shape)))
print('eigen_value :', pca_SD.explained_variance_)
print('explained variance ratio :', pca_SD.explained_variance_ratio_)
print('cum_explained variance ratio :', sum(pca_SD.explained_variance_ratio_))
# plt.figure(figsize = (10, 7))
# plt.plot(np.cumsum(pca_SD.explained_variance_ratio_), 'o-')
# plt.title('cumulative explained variance')
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.figure(figsize = (10, 7))
# plt.plot([1,2,3,4,5,6,7,8,9,10], pca_SD.explained_variance_ratio_, 'o-')
# plt.title('pca_SD Scree Plot')
# plt.xlabel('Number of Dimensions')
# plt.ylabel('Eigenvalue');

# RB
pca_RB = PCA(n_components = 3).fit(X_train_RBsc)
X_train_pca_RB = pca_RB.transform(X_train_RBsc)
X_val_pca_RB = pca_RB.transform(X_val_RBsc)
X_test_pca_RB = pca_RB.transform(X_test_RBsc)
print("Original shape: {}".format(str(X_train_RBsc.shape)))
print("Reduced shape: {}".format(str(X_train_pca_RB.shape)))
print('eigen_value :', pca_RB.explained_variance_)
print('explained variance ratio :', pca_RB.explained_variance_ratio_)
print('cum_explained variance ratio :', sum(pca_RB.explained_variance_ratio_))
# plt.figure(figsize = (10, 7))
# plt.plot(np.cumsum(pca_RB.explained_variance_ratio_), 'o-')
# plt.title('cumulative explained variance')
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.figure(figsize = (10, 7))
# plt.plot([1,2,3,4,5,6,7,8,9,10], pca_RB.explained_variance_ratio_, 'o-')
# plt.title('pca_RB Scree Plot')
# plt.xlabel('Number of Dimensions')
# plt.ylabel('Eigenvalue');

''' # 시각화 ''' 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (10, 8))
ax = Axes3D(fig, elev = 38, azim = 110)
ax.scatter(X_test_pca_SD[:, 0], X_test_pca_SD[:, 1], X_test_pca_SD[:, 2], c = y_test,
           cmap = plt.cm.Set1, edgecolor = 'k', s = 28)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()

# # GIF 이미지화
# for ii in np.arange(0, 360, 1):
#     ax.view_init(elev = 38, azim = ii)
#     fig.savefig('gif_image%d.png' % ii)

#%% 판별 분석
'''# 선형판별분석법(linear discriminant analysis, LDA) '''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
# SD
cld_SD = LinearDiscriminantAnalysis(store_covariance = True).fit(X_train_SDsc, y_train)
y_train_pred = cld_SD.predict(X_train_SDsc)
y_val_pred = cld_SD.predict(X_val_SDsc)
y_test_pred = cld_SD.predict(X_test_SDsc)
print("SD선형 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.9109
print("SD선형 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')	          # 0.9279

# RB
cld_RB = LinearDiscriminantAnalysis(store_covariance = True).fit(X_train_RBsc, y_train)
y_train_pred = cld_RB.predict(X_train_RBsc)
y_val_pred = cld_RB.predict(X_val_RBsc)
y_test_pred = cld_RB.predict(X_test_RBsc)
print("RB선형 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.9109
print("RB선형 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')	          # 0.9279

# no-scaler
cld = LinearDiscriminantAnalysis(store_covariance = True).fit(X_train, y_train)
y_train_pred = cld.predict(X_train)
y_val_pred = cld.predict(X_val)
y_test_pred = cld.predict(X_test)
print("선형 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.9109
print("선형 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')          # 0.9279

# pca_SD
cld_pca_SD = LinearDiscriminantAnalysis(store_covariance = True).fit(X_train_pca_SD, y_train)
y_train_pred = cld_pca_SD.predict(X_train_pca_SD)
y_val_pred = cld_pca_SD.predict(X_val_pca_SD)
y_test_pred = cld_pca_SD.predict(X_test_pca_SD)
print("pca_SD선형 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.8225
print("pca_SD선형 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')          # 0.8256

# pca_RB
cld_pca_RB = LinearDiscriminantAnalysis(store_covariance = True).fit(X_train_pca_RB, y_train)
y_train_pred = cld_pca_RB.predict(X_train_pca_RB)
y_val_pred = cld_pca_RB.predict(X_val_pca_RB)
y_test_pred = cld_pca_RB.predict(X_test_pca_RB)
print("pca_RB선형 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.8101
print("pca_RB선형 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')          # 0.8349

''' # 이차판별분석법(quadratic discriminant analysis, QDA) '''
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
# SD
cqd_SD = QuadraticDiscriminantAnalysis(store_covariance = True).fit(X_train_SDsc, y_train)
y_train_pred = cqd_SD.predict(X_train_SDsc)
y_val_pred = cqd_SD.predict(X_val_SDsc)
y_test_pred = cqd_SD.predict(X_test_SDsc)
print("SD이차 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.8953
print("SD이차 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')	          # 0.8907

# RB
cqd_RB = QuadraticDiscriminantAnalysis(store_covariance = True).fit(X_train_RBsc, y_train)
y_train_pred = cqd_RB.predict(X_train_SDsc)
y_val_pred = cqd_RB.predict(X_val_SDsc)
y_test_pred = cqd_RB.predict(X_test_SDsc)
print("RB이차 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.8357
print("RB이차 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')	          # 0.8302

# no-scaler
cqd = QuadraticDiscriminantAnalysis(store_covariance = True).fit(X_train, y_train)
y_train_pred = cqd.predict(X_train)
y_val_pred = cqd.predict(X_val)
y_test_pred = cqd.predict(X_test)
print("이차 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.8953
print("이차 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')          # 0.8907

# pca_SD
cqd_pca_SD = QuadraticDiscriminantAnalysis(store_covariance = True).fit(X_train_pca_SD, y_train)
y_train_pred = cqd_pca_SD.predict(X_train_pca_SD)
y_val_pred = cqd_pca_SD.predict(X_val_pca_SD)
y_test_pred = cqd_pca_SD.predict(X_test_pca_SD)
print("pca_SD이차 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.8171
print("pca_SD이차 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')          # 0.8326

# pca_RB
cqd_pca_RB = QuadraticDiscriminantAnalysis(store_covariance = True).fit(X_train_pca_RB, y_train)
y_train_pred = cqd_pca_RB.predict(X_train_pca_RB)
y_val_pred = cqd_pca_RB.predict(X_val_pca_RB)
y_test_pred = cqd_pca_RB.predict(X_test_pca_RB)
print("pca_RB이차 train:", f'{accuracy_score(y_train, y_train_pred):.4f}')    # 0.8116
print("pca_RB이차 val:", f'{accuracy_score(y_val, y_val_pred):.4f}')          # 0.8302

''' # 최종모형 '''

#%% 로지스틱 회귀분석
''' # 각각으로 그리드 서치 '''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# SD                 데이터가 적어서 lbgs보다 liblinear가 더 좋은 결과 나옴
classifier = LogisticRegression(max_iter = 2000, random_state = 0)
h_para = [{'C': np.logspace(-4, 4, 20), 'penalty': [ 'l1', 'l2'], 'solver': ['liblinear']},
          {'C': np.logspace(-4, 4, 20), 'penalty': ['l2'], 'solver': ['lbfgs']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_SDsc, y_train)
best_accuracy_SD = grid_search.best_score_
best_parameters_SD = grid_search.best_params_
print(best_parameters_SD, f'{best_accuracy_SD:.4f}')
# {'C': 4.281332398719396, 'penalty': 'l2', 'solver': 'liblinear'} 0.9699

# RB
classifier = LogisticRegression(max_iter = 2000, random_state = 0)
h_para = [{'C': np.logspace(-4, 4, 20), 'penalty': [ 'l1', 'l2'], 'solver': ['liblinear']},
          {'C': np.logspace(-4, 4, 20), 'penalty': ['l2'], 'solver': ['lbfgs']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_RBsc, y_train)
best_accuracy_RB = grid_search.best_score_
best_parameters_RB = grid_search.best_params_
print(best_parameters_RB, f'{best_accuracy_RB:.4f}')
# {'C': 78.47599703514607, 'penalty': 'l2', 'solver': 'liblinear'} 0.9699

# no-scaler
classifier = LogisticRegression(max_iter = 1000000)
h_para = [{'C': np.logspace(-4, 4, 20), 'penalty': [ 'l1', 'l2'], 'solver': ['liblinear']},
          {'C': np.logspace(-4, 4, 20), 'penalty': ['l2'], 'solver': ['lbfgs']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_parameters, f'{best_accuracy:.4f}')
# {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'} 0.9697

# pca_SD
classifier = LogisticRegression(max_iter = 2000, random_state = 0)
h_para = [{'C': np.logspace(-4, 4, 20), 'penalty': [ 'l1', 'l2'], 'solver': ['liblinear']},
          {'C': np.logspace(-4, 4, 20), 'penalty': ['l2'], 'solver': ['lbfgs']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_pca_SD, y_train)
best_accuracy_pca_SD = grid_search.best_score_
best_parameters_pca_SD = grid_search.best_params_
print(best_parameters_pca_SD, f'{best_accuracy_pca_SD:.4f}')
# {'C': 4.281332398719396, 'penalty': 'l1', 'solver': 'liblinear'} 0.8969

# pca_RB
classifier = LogisticRegression(max_iter = 2000, random_state = 0)
h_para = [{'C': np.logspace(-4, 4, 20), 'penalty': [ 'l1', 'l2'], 'solver': ['liblinear']},
          {'C': np.logspace(-4, 4, 20), 'penalty': ['l2'], 'solver': ['lbfgs']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_pca_RB, y_train)
best_accuracy_pca_RB = grid_search.best_score_
best_parameters_pca_RB = grid_search.best_params_
print(best_parameters_pca_RB, f'{best_accuracy_pca_RB:.4f}')
# {'C': 0.08858667904100823, 'penalty': 'l2', 'solver': 'liblinear'} 0.8910

''' # grid search의 best_parameters에 대한 결과치 '''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# SD
lr = LogisticRegression(C = 4.28, penalty = 'l2', solver = 'liblinear').fit(X_train_SDsc, y_train)
print("SD_lr train: {:.4f}".format(lr.score(X_train_SDsc, y_train)))    # 0.9178
print("SD_lr val: {:.4f}".format(lr.score(X_val_SDsc, y_val)))          # 0.9256

# y_train_pred = lr.predict(X_train_SDsc)      
# y_val_pred = lr.predict(X_val_SDsc)  
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    #train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    #val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize = (5,5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

# RB
lr = LogisticRegression(C = 78.47, penalty = 'l2', solver = 'liblinear').fit(X_train_RBsc, y_train)
print("RB_lr train: {:.4f}".format(lr.score(X_train_RBsc, y_train)))    # 0.9178
print("BR_lr val: {:.4f}".format(lr.score(X_val_RBsc, y_val)))          # 0.9233

# y_train_pred = lr.predict(X_train_RBsc)      
# y_val_pred = lr.predict(X_val_RBsc)  
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    #train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    #val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize = (5,5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

# no-scaler
lr = LogisticRegression(C = 10, penalty = 'l1', solver = 'liblinear', max_iter = 200).fit(X_train, y_train)
print("lr train: {:.4f}".format(lr.score(X_train, y_train)))            # 0.9171
print("lr val: {:.4f}".format(lr.score(X_val, y_val)))                  # 0.9233

# y_train_pred = lr.predict(X_train)
# y_val_pred = lr.predict(X_val)
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    #train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    #val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize = (5,5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

# pca_SD
lr = LogisticRegression(C = 4.28, penalty = 'l1', solver = 'liblinear').fit(X_train_pca_SD, y_train)
print("pca_SD_lr train: {:.4f}".format(lr.score(X_train_pca_SD, y_train)))            # 0.8209
print("pca_SD_lr val: {:.4f}".format(lr.score(X_val_pca_SD, y_val)))                  # 0.8302

# y_train_pred = lr.predict(X_train_pca_SD)
# y_val_pred = lr.predict(X_val_pca_SD)
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    #train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    #val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize = (5,5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

# pca_RB
lr = LogisticRegression(C = 0.08, penalty = 'l2', solver = 'lbfgs').fit(X_train_pca_RB, y_train)
print("pca_RB_lr train: {:.4f}".format(lr.score(X_train_pca_RB, y_train)))            # 0.8109
print("pca_RB_lr val: {:.4f}".format(lr.score(X_val_pca_RB, y_val)))                  # 0.8326

# y_train_pred = lr.predict(X_train_pca_RB)
# y_val_pred = lr.predict(X_val_pca_RB)
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    #train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    #val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize = (5,5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

''' # 최종 모형 '''
import statsmodels.api as sm
y_traint = list(y_train)
logit = sm.Logit(y_traint, x1)
result= logit.fit()
print(result.summary())
print(np.exp(result.params))

''' # 시각화 '''
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_v = X_test_pca_SD
Y_v = y_test
X_v = X_v[np.logical_or(Y_v==0,Y_v==1)]
Y_v = Y_v[np.logical_or(Y_v==0,Y_v==1)]

clf = LogisticRegression(C = 3, penalty = 'l2', solver = 'liblinear').fit(X_v, Y_v)
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure(figsize = (8, 7))
ax  = fig.add_subplot(111, projection = '3d')
ax.plot3D(X_v[Y_v==0,0], X_v[Y_v==0,1], X_v[Y_v==0,2],'ob')
ax.plot3D(X_v[Y_v==1,0], X_v[Y_v==1,1], X_v[Y_v==1,2],'sr')
ax.plot_surface(x, y, z(x,y))
ax.view_init(32, 222)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()

# for ii in np.arange(0, 360, 1):
#     ax.view_init(elev=32, azim=ii)
#     fig.savefig('gif_image%d.png' % ii)


#%% KNN 분석
''' # 각각으로 그리드 서치 '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# SD
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100), 'weights': ['uniform', 'distance']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_SDsc, y_train)
best_accuracy_SD = grid_search.best_score_
best_parameters_SD = grid_search.best_params_
print(best_parameters_SD, f'{best_accuracy_SD:.4f}')     # {'n_neighbors': 21, 'weights': 'distance'} 0.9463

# RB
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100), 'weights': ['uniform', 'distance']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_RBsc, y_train)
best_accuracy_RB = grid_search.best_score_
best_parameters_RB = grid_search.best_params_
print(best_parameters_RB, f'{best_accuracy_RB:.4f}')     # {'n_neighbors': 16, 'weights': 'distance'} 0.9441

# no-scaler
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100), 'weights': ['uniform', 'distance']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_parameters, f'{best_accuracy:.4f}')       # {'n_neighbors': 17, 'weights': 'distance'} 0.7519

# pca_SD
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100), 'weights': ['uniform', 'distance']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_pca_SD, y_train)
best_accuracy_pca_SD = grid_search.best_score_
best_parameters_pca_SD = grid_search.best_params_
print(best_parameters_pca_SD, f'{best_accuracy_pca_SD:.4f}')    # {'n_neighbors': 43, 'weights': 'uniform'} 0.8969

# pca_RB
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100), 'weights': ['uniform', 'distance']}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_pca_RB, y_train)
best_accuracy_pca_RB = grid_search.best_score_
best_parameters_pca_RB = grid_search.best_params_
print(best_parameters_pca_RB, f'{best_accuracy_pca_RB:.4f}')    # {'n_neighbors': 35, 'weights': 'distance'} 0.8926

''' # grid search의 best_parameters에 대한 결과치 '''
from sklearn.neighbors import KNeighborsClassifier
# SD
KNN = KNeighborsClassifier(n_neighbors = 21, weights = 'distance').fit(X_train_SDsc, y_train)
print("KNN_SD train: {:.4f}".format(KNN.score(X_train_SDsc, y_train)))     # 1.0000
print("KNN_SD val: {:.4f}".format(KNN.score(X_val_SDsc, y_val)))           # 0.9023

# RB
KNN = KNeighborsClassifier(n_neighbors = 16, weights = 'distance').fit(X_train_RBsc, y_train)
print("KNN_RB train: {:.4f}".format(KNN.score(X_train_RBsc, y_train)))     # 1.0000
print("KNN_RB val: {:.4f}".format(KNN.score(X_val_RBsc, y_val)))           # 0.9047

# no-scaler
KNN = KNeighborsClassifier(n_neighbors = 17, weights = 'distance').fit(X_train, y_train)
print("KNN train: {:.4f}".format(KNN.score(X_train, y_train)))                # 1.0000
print("KNN val: {:.4f}".format(KNN.score(X_val, y_val)))                      # 0.6744

# pca_SD
KNN = KNeighborsClassifier(n_neighbors = 43, weights = 'uniform').fit(X_train_pca_SD, y_train)
print("KNN train: {:.4f}".format(KNN.score(X_train_pca_SD, y_train)))                # 0.8264
print("KNN val: {:.4f}".format(KNN.score(X_val_pca_SD, y_val)))                      # 0.8256

# pca_RB
KNN = KNeighborsClassifier(n_neighbors = 35, weights = 'distance').fit(X_train_pca_RB, y_train)
print("KNN train: {:.4f}".format(KNN.score(X_train_pca_RB, y_train)))                # 1.0000
print("KNN val: {:.4f}".format(KNN.score(X_val_pca_RB, y_val)))                      # 0.8302

''' # 과적합 발생 -> 직접 찾기로함 '''
train_accuracy = []
val_accuracy = []
neighbors_settings = range(3, 100)
from sklearn.neighbors import KNeighborsClassifier
for n_neighbors in neighbors_settings:
    knMod = KNeighborsClassifier(n_neighbors = n_neighbors, weights = 'distance').fit(X_train_SDsc, y_train)
    train_accuracy.append(knMod.score(X_train_SDsc, y_train))
    val_accuracy.append(knMod.score(X_val_SDsc, y_val))
plt.plot(neighbors_settings, train_accuracy, label = "train accuracy")
plt.plot(neighbors_settings, val_accuracy, label = "val accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

''' # weights = 'distance'하면 과적합 발생 weights = 'uniform' 다시 그리드 서치 실행 '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# SD
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100)}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_SDsc, y_train)
best_accuracy_SD = grid_search.best_score_
best_parameters_SD = grid_search.best_params_
print(best_parameters_SD, f'{best_accuracy_SD:.4f}')        # {'n_neighbors': 21} 0.9456

# RB
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100)}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_RBsc, y_train)
best_accuracy_RB = grid_search.best_score_
best_parameters_RB = grid_search.best_params_
print(best_parameters_RB, f'{best_accuracy_RB:.4f}')        # {'n_neighbors': 16} 0.9432

# no-scaler
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100)}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_parameters, f'{best_accuracy:.4f}')      # {'n_neighbors': 17} 0.7484

# pca_SD
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100)}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_pca_SD, y_train)
best_accuracy_pca_SD = grid_search.best_score_
best_parameters_pca_SD = grid_search.best_params_
print(best_parameters_pca_SD, f'{best_accuracy_pca_SD:.4f}')    # {'n_neighbors': 43} 0.8969

# pca_RB
classifier = KNeighborsClassifier()
h_para = [{'n_neighbors': range(1, 100)}]
grid_search = GridSearchCV(estimator = classifier, param_grid = h_para, scoring = 'roc_auc', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train_pca_RB, y_train)
best_accuracy_pca_RB = grid_search.best_score_
best_parameters_pca_RB = grid_search.best_params_
print(best_parameters_pca_RB, f'{best_accuracy_pca_RB:.4f}')    # {'n_neighbors': 35} 0.8920

''' # grid search의 best_parameters에 대한 결과치 '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
# SD
KNN = KNeighborsClassifier(n_neighbors = 21).fit(X_train_SDsc, y_train)
print("KNN_SD train: {:.4f}".format(KNN.score(X_train_SDsc, y_train)))     # 0.8798
print("KNN_SD val: {:.4f}".format(KNN.score(X_val_SDsc, y_val)))           # 0.9047

# y_train_pred = KNN.predict(X_train_SDsc)
# y_val_pred = KNN.predict(X_val_SDsc)
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    # train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    # val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize=(5,5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

# RB
KNN = KNeighborsClassifier(n_neighbors = 16).fit(X_train_RBsc, y_train)
print("KNN_RB train: {:.4f}".format(KNN.score(X_train_RBsc, y_train)))     # 0.8876
print("KNN_RB val: {:.4f}".format(KNN.score(X_val_RBsc, y_val)))           # 0.9116

# y_train_pred = KNN.predict(X_train_RBsc)
# y_val_pred = KNN.predict(X_val_RBsc)
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    # train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    # val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize=(5,5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

# no-scaler
KNN = KNeighborsClassifier(n_neighbors = 17).fit(X_train, y_train)
print("KNN train: {:.4f}".format(KNN.score(X_train, y_train)))                # 0.7349
print("KNN val: {:.4f}".format(KNN.score(X_val, y_val)))                      # 0.6791

# y_train_pred = KNN.predict(X_train)
# y_val_pred = KNN.predict(X_val)
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    # train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    # val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize=(5,5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

# pca_SD
KNN = KNeighborsClassifier(n_neighbors = 43).fit(X_train_pca_SD, y_train)
print("KNN train: {:.4f}".format(KNN.score(X_train_pca_SD, y_train)))                # 0.8264
print("KNN val: {:.4f}".format(KNN.score(X_val_pca_SD, y_val)))                      # 0.8256

# y_train_pred = KNN.predict(X_train_pca_SD)
# y_val_pred = KNN.predict(X_val_pca_SD)
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    # train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    # val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize = (5, 5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

# pca_RB
KNN = KNeighborsClassifier(n_neighbors = 35).fit(X_train_pca_RB, y_train)
print("KNN train: {:.4f}".format(KNN.score(X_train_pca_RB, y_train)))                # 0.8178
print("KNN val: {:.4f}".format(KNN.score(X_val_pca_RB, y_val)))                      # 0.8349

# y_train_pred = KNN.predict(X_train_pca_RB)
# y_val_pred = KNN.predict(X_val_pca_RB)
# print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum())
# print('Misclassified test samples: %d' %(y_val!=y_val_pred).sum()) 
# conf1 = confusion_matrix(y_true = y_train, y_pred = y_train_pred)
# print(conf1)    # train 오분류
# conf2 = confusion_matrix(y_true = y_val, y_pred = y_val_pred) 
# print(conf2)    # val 오분류
# y_true = y_val
# cm = confusion_matrix(y_true, y_val_pred)
# f, ax = plt.subplots(figsize = (5, 5))
# sns.heatmap(cm, annot = True, linewidths = 0.5, fmt = '.0f', ax = ax )
# plt.xlabel('Negative')
# plt.ylabel('Positive')
# plt.show()

''' # 최종 모형 '''
from sklearn.neighbors import KNeighborsClassifier


# # GIF 이미지화
# for ii in np.arange(0, 360, 1):
#     ax.view_init(elev = 38, azim = ii)
#     fig.savefig('gif_image%d.png' % ii)




#%% Decision Tree
''' # 각각으로 랜덤 서치 '''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
# SD
classifier = DecisionTreeClassifier(random_state = 0)
h_para = [{'criterion': ['entropy', 'gini'], 'max_depth': np.arange(1, 20),
           'max_leaf_nodes': np.arange(2, 20), 'min_samples_split': np.arange(2, 20),
           'min_samples_leaf': np.arange(1, 20)}]
rds = RandomizedSearchCV(classifier, param_distributions = h_para, n_iter = 10,
                         cv = 10, scoring = 'roc_auc', n_jobs = -1).fit(X_train_SDsc, y_train)
best_accuracy_SD = rds.best_score_
best_parameters_SD = rds.best_params_
print(best_parameters_SD, f'{best_accuracy_SD:.4f}')
# {'min_samples_split': 11, 'min_samples_leaf': 10, 'max_leaf_nodes': 12, 
# 'max_depth': 14, 'criterion': 'entropy'} 0.9264

# RB
classifier = DecisionTreeClassifier(random_state = 0)
h_para = [{'criterion': ['entropy', 'gini'], 'max_depth': np.arange(1, 20),
           'max_leaf_nodes': np.arange(2, 20), 'min_samples_split': np.arange(2, 20),
           'min_samples_leaf': np.arange(1, 20)}]
rds = RandomizedSearchCV(classifier, param_distributions = h_para, n_iter = 10,
                         cv = 10, scoring = 'roc_auc', n_jobs = -1).fit(X_train_RBsc, y_train)
best_accuracy_RB = rds.best_score_
best_parameters_RB = rds.best_params_
print(best_parameters_RB, f'{best_accuracy_RB:.4f}')
# {'min_samples_split': 13, 'min_samples_leaf': 14, 'max_leaf_nodes': 17, 
# 'max_depth': 17, 'criterion': 'entropy'} 0.9238

# pca_SD
classifier = DecisionTreeClassifier(random_state = 0)
h_para = [{'criterion': ['entropy', 'gini'], 'max_depth': np.arange(1, 20),
           'max_leaf_nodes': np.arange(2, 20), 'min_samples_split': np.arange(2, 20),
           'min_samples_leaf': np.arange(1, 20)}]
rds = RandomizedSearchCV(classifier, param_distributions = h_para, n_iter = 10,
                         cv = 10, scoring = 'roc_auc', n_jobs = -1).fit(X_train_pca_RB, y_train)
best_accuracy_pca_SD = rds.best_score_
best_parameters_pca_SD = rds.best_params_
print(best_parameters_pca_SD, f'{best_accuracy_pca_SD:.4f}')
# {'min_samples_split': 10, 'min_samples_leaf': 16, 'max_leaf_nodes': 19, 
#  'max_depth': 15, 'criterion': 'entropy'} 0.8598

# pca_RB
classifier = DecisionTreeClassifier(random_state = 0)
h_para = [{'criterion': ['entropy', 'gini'], 'max_depth': np.arange(1, 20),
           'max_leaf_nodes': np.arange(2, 20), 'min_samples_split': np.arange(2, 20),
           'min_samples_leaf': np.arange(1, 20)}]
rds = RandomizedSearchCV(classifier, param_distributions = h_para, n_iter = 10,
                         cv = 10, scoring = 'roc_auc', n_jobs = -1).fit(X_train_pca_RB, y_train)
best_accuracy_pca_RB = rds.best_score_
best_parameters_pca_RB = rds.best_params_
print(best_parameters_pca_RB, f'{best_accuracy_pca_RB:.4f}')
# {'min_samples_split': 11, 'min_samples_leaf': 12, 'max_leaf_nodes': 15, 
# 'max_depth': 7, 'criterion': 'entropy'} 0.8538

''' # grid search의 best_parameters에 대한 결과치 '''
from sklearn.tree import DecisionTreeClassifier
# SD
DTC_SD = DecisionTreeClassifier(max_depth = 14, max_leaf_nodes = 12, min_samples_leaf = 10, 
                                min_samples_split = 11, criterion = 'entropy').fit(X_train_SDsc, y_train)
print("DTC_SD train: {:.4f}".format(DTC_SD.score(X_train_SDsc, y_train)))     # 0.8822
print("DTC_SD val: {:.4f}".format(DTC_SD.score(X_val_SDsc, y_val)))           # 0.8930

# RB
DTC_RB = DecisionTreeClassifier(max_depth = 17, max_leaf_nodes = 17, min_samples_leaf = 14, 
                                min_samples_split = 13, criterion = 'entropy').fit(X_train_RBsc, y_train)
print("DTC_RB train: {:.4f}".format(DTC_SD.score(X_train_RBsc, y_train)))     # 0.7171
print("DTC_RB val: {:.4f}".format(DTC_SD.score(X_val_RBsc, y_val)))           # 0.7116

# pca_SD
DTC_pca_SD = DecisionTreeClassifier(max_depth = 15, max_leaf_nodes = 19, min_samples_leaf = 16,
                                    min_samples_split = 10, criterion = 'entropy').fit(X_train_pca_SD, y_train)
print("DTC_pca_SD train: {:.4f}".format(DTC_pca_SD.score(X_train_pca_SD, y_train)))           # 0.8295
print("DTC_pca_SD val: {:.4f}".format(DTC_pca_SD.score(X_val_pca_SD, y_val)))                 # 0.8047

# pca_RB
DTC_pca_RB = DecisionTreeClassifier(max_depth = 7, max_leaf_nodes = 15, min_samples_leaf = 12,
                                    min_samples_split = 11, criterion = 'entropy').fit(X_train_pca_RB, y_train)
print("DTC_pca_RB train: {:.4f}".format(DTC_pca_RB.score(X_train_pca_RB, y_train)))           # 0.8101
print("DTC_pca_RB val: {:.4f}".format(DTC_pca_RB.score(X_val_pca_RB, y_val)))                 # 0.8093

''' # 최종 모형 '''
from sklearn.tree import DecisionTreeClassifier
DTC_SD = DecisionTreeClassifier(max_depth = 4, max_leaf_nodes = 10).fit(X_train_SDsc, y_train)
print("DTC_SD train: {:.4f}".format(DTC_SD.score(X_train_SDsc, y_train)))     # 0.8946
print("DTC_SD val: {:.4f}".format(DTC_SD.score(X_val_SDsc, y_val)))           # 0.9070
print("DTC_SD test: {:.4f}".format(DTC_SD.score(X_test_SDsc, y_test)))        # 0.8884

''' # 시각화 '''
from pydotplus import graph_from_dot_data  
from sklearn.tree import export_graphviz  
from IPython.display import Image

dot_data1 = export_graphviz(DTC_SD, filled = True, rounded = True, class_names = ["Lose","Win"], 
                            feature_names = X_train.columns[0:], out_file = None)
graph1 = graph_from_dot_data(dot_data1)
graph1.write_png('tree1.png')
Image(graph1.create_png())

dot_data4 = export_graphviz(DTC_pca_SD, filled = True, rounded = True, class_names = ["Lose","Win"], 
                            out_file = None)
graph4 = graph_from_dot_data(dot_data4)
graph4.write_png('tree4.png') 
Image(graph4.create_png())

dot_data5 = export_graphviz(DTC_pca_RB, filled = True, rounded = True, class_names = ["Lose","Win"], 
                            out_file = None)
graph5 = graph_from_dot_data(dot_data5)
graph5.write_png('tree5.png') 
Image(graph5.create_png())

''' # 트리의 특성 중요도 '''
print("특성 중요도:\n", DTC_SD.feature_importances_)
def plot_feature_importances_credit(model):
    n_features = X_train_SDsc.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), X_train.columns[0:])
    plt.xlabel("feature_importance")
    plt.ylabel("feature")
    plt.ylim(-1,n_features)
plot_feature_importances_credit(DTC_SD)

print("특성 중요도:\n", DTC_RB.feature_importances_)
def plot_feature_importances_credit(model):
    n_features = X_train_RBsc.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), X_train.columns[0:])
    plt.xlabel("feature_importance")
    plt.ylabel("feature")
    plt.ylim(-1,n_features)
plot_feature_importances_credit(DTC_RB)

print("특성 중요도:\n", DTC_pca_SD.feature_importances_)
def plot_feature_importances_credit(model):
    n_features = X_train_pca_SD.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), X_train.columns[0:])
    plt.xlabel("feature_importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
plot_feature_importances_credit(DTC_pca_SD)

print("특성 중요도:\n", DTC_pca_RB.feature_importances_)
def plot_feature_importances_credit(model):
    n_features = X_train_pca_RB.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), X_train.columns[0:])
    plt.xlabel("feature_importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
plot_feature_importances_credit(DTC_pca_RB)

#%% SVM(Support vector machine)
''' # 각각으로 랜덤 서치 (grid search 하기에는 시간이 너무 오래걸림) '''
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
# C값이 슬랙변수(높으면 언더피팅, 낮으면 오버피팅), gamma (높으면 오버피팅, 낮으면 언더피팅)
# SD
classifier = SVC()
h_para = [{'kernel': [ 'linear' ], 'C': np.logspace(-3, 2, 6)},
          {'kernel': ['rbf'], 'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}]
random_search = RandomizedSearchCV(classifier, param_distributions = h_para, n_iter = 10,
                                   cv = 10, scoring = 'roc_auc', n_jobs = -1)
rds = random_search.fit(X_train_SDsc, y_train)
best_accuracy_rds_SD = rds.best_score_
best_parameters_rds_SD = rds.best_params_
print(best_parameters_rds_SD, f'{best_accuracy_rds_SD:.4f}')  # {'kernel': 'linear', 'C': 1.0} 0.9692

# RB
classifier = SVC()
h_para = [{'kernel': [ 'linear' ], 'C': np.logspace(-3, 2, 6)},
          {'kernel': ['rbf'], 'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}]
random_search = RandomizedSearchCV(classifier, param_distributions = h_para, n_iter = 10,
                                   cv = 10, scoring = 'roc_auc', n_jobs = -1)
rds = random_search.fit(X_train_SDsc, y_train)
best_accuracy_rds_RB = rds.best_score_
best_parameters_rds_RB = rds.best_params_
print(best_parameters_rds_RB, f'{best_accuracy_rds_RB:.4f}')  # {'kernel': 'rbf', 'gamma': 0.01, 'C': 100.0} 0.9648

# pca_SD
classifier = SVC()
h_para = [{'kernel': [ 'linear' ], 'C': np.logspace(-3, 2, 6)},
          {'kernel': ['rbf'], 'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}]
random_search = RandomizedSearchCV(classifier, param_distributions = h_para, n_iter = 10,
                                   cv = 10, scoring = 'roc_auc', n_jobs = -1)
rds = random_search.fit(X_train_pca_SD, y_train)
best_accuracy_rds_pca_SD = rds.best_score_
best_parameters_rds_pca_SD = rds.best_params_
print(best_parameters_rds_pca_SD, f'{best_accuracy_rds_pca_SD:.4f}')   # {'kernel': 'linear', 'C': 10.0} 0.8968

# pca_RB
classifier = SVC()
h_para = [{'kernel': [ 'linear' ], 'C': np.logspace(-3, 2, 6)},
          {'kernel': ['rbf'], 'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}]
random_search = RandomizedSearchCV(classifier, param_distributions = h_para, n_iter = 10,
                                   cv = 10, scoring = 'roc_auc', n_jobs = -1)
rds = random_search.fit(X_train_pca_RB, y_train)
best_accuracy_rds_pca_RB = rds.best_score_
best_parameters_rds_pca_RB = rds.best_params_
print(best_parameters_rds_pca_RB, f'{best_accuracy_rds_pca_RB:.4f}')   # {'kernel': 'linear', 'C': 1.0} 0.8909

''' # random search의 best_parameters에 대한 결과치 '''
from sklearn.svm import SVC
# SD
svm_SD = SVC(kernel = 'linear', C = 1).fit(X_train_SDsc, y_train)
print("svm_SD train: {:.4f}".format(svm_SD.score(X_train_SDsc, y_train)))    # 0.9233
print("svm_SD val: {:.4f}".format(svm_SD.score(X_val_SDsc, y_val)))          # 0.9279

# RB
svm_RB = SVC(kernel = 'rbf', C = 100, gamma = 0.01).fit(X_train_RBsc, y_train)
print("svm_RB train: {:.4f}".format(svm_RB.score(X_train_RBsc, y_train)))    # 0.9388
print("svm_RB val: {:.4f}".format(svm_RB.score(X_val_RBsc, y_val)))          # 0.9302

# pca_SD
svm_pca_SD = SVC(kernel = 'linear', C = 10).fit(X_train_pca_SD, y_train)
print("svm_pca_SD train: {:.4f}".format(svm_pca_SD.score(X_train_pca_SD, y_train)))    # 0.8202
print("svm_pca_SD val: {:.4f}".format(svm_pca_SD.score(X_val_pca_SD, y_val)))          # 0.8279

# pca_RB
svm_pca_RB = SVC(kernel = 'linear', C = 1).fit(X_train_pca_RB, y_train)
print("svm_pca_RB train: {:.4f}".format(svm_pca_RB.score(X_train_pca_RB, y_train)))    # 0.8116
print("svm_pca_RB val: {:.4f}".format(svm_pca_RB.score(X_val_pca_RB, y_val)))          # 0.8256

''' # 최종 모형 '''
from sklearn.svm import SVC
svm_SD = SVC(kernel = 'linear', C = 1).fit(X_train_SDsc, y_train)
print("svm_SD train: {:.4f}".format(svm_SD.score(X_train_SDsc, y_train)))    # 0.9233
print("svm_SD val: {:.4f}".format(svm_SD.score(X_val_SDsc, y_val)))          # 0.9279
print("svm_SD test: {:.4f}".format(svm_SD.score(X_test_SDsc, y_test)))       # 0.9093

''' # 시각화 '''
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_svm = X_test_pca_SD  # we only take the first three features.
Y_svm = y_test
X_svm = X_svm[np.logical_or(Y_svm == 0, Y_svm == 1)]
Y_svm = Y_svm[np.logical_or(Y_svm == 0,Y_svm == 1)]

clf = SVC(kernel = 'linear', C = 1).fit(X_svm, Y_svm)
z_svm = lambda x_svm, y_svm:\
    (-clf.intercept_[0]-clf.coef_[0][0]*x_svm-clf.coef_[0][1]*y_svm)/clf.coef_[0][2]

tmp = np.linspace(-5, 5, 100)
x_svm, y_svm = np.meshgrid(tmp, tmp)

fig = plt.figure(figsize = (10, 8))
ax  = fig.add_subplot(111, projection = '3d')
ax.plot3D(X_svm[Y_svm == 0,0], X_svm[Y_svm == 0,1], X_svm[Y_svm == 0,2], 'ob')
ax.plot3D(X_svm[Y_svm == 1,0], X_svm[Y_svm == 1,1], X_svm[Y_svm == 1,2], 'sr')
ax.plot_surface(x_svm, y_svm, z_svm(x_svm, y_svm))
ax.view_init(38, 110)
ax.set_title("SVM")
ax.set_xlabel("1st ")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd ")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd ")
ax.w_zaxis.set_ticklabels([])
plt.show()

# # GIF 이미지화
# for ii in np.arange(0, 360, 1):
#     ax.view_init(elev = 38, azim = ii)
#     fig.savefig('gif_image%d.png' % ii)

# from sklearn.metrics import plot_confusion_matrix, accuracy_score
# svm = SVC(kernel='linear', C=1).fit(X_train_SDsc, y_train)
# matrix = plot_confusion_matrix(svm, X_test, y_test,
#                                  cmap=plt.cm.Blues)
# plt.title('Confusion matrix for linear SVM')
# plt.show(matrix)
# plt.show()

# 시각화
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X_svm = X_test_SDsc  # we only take the first three features.
Y_svm = y_test
X_svm = X_svm[np.logical_or(Y_svm == 0, Y_svm == 1)]
Y_svm = Y_svm[np.logical_or(Y_svm == 0,Y_svm == 1)]

clf = SVC(kernel = 'linear', C = 1).fit(X_svm, Y_svm)
z_svm = lambda x_svm, y_svm:\
    (-clf.intercept_[0]-clf.coef_[0][0]*x_svm-clf.coef_[0][1]*y_svm)/clf.coef_[0][2]

tmp = np.linspace(-2, 5, 100)
x_svm, y_svm = np.meshgrid(tmp, tmp)

fig = plt.figure(figsize = (10, 8))
ax  = fig.add_subplot(111, projection = '3d')
ax.plot3D(X_svm[Y_svm == 0,0], X_svm[Y_svm == 0,1], X_svm[Y_svm == 0,2], 'ob')
ax.plot3D(X_svm[Y_svm == 1,0], X_svm[Y_svm == 1,1], X_svm[Y_svm == 1,2], 'sr')
ax.plot_surface(x_svm, y_svm, z_svm(x_svm, y_svm))
ax.view_init(38, 110)
ax.set_title("SVM")
ax.set_xlabel("1st ")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd ")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd ")
ax.w_zaxis.set_ticklabels([])
plt.show()

#%% 실제 데이터로 예측
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

patht = r'C:\Users\USER\Documents\port\res_pre.csv'
new_data = pd.read_csv(patht)
X1 = new_data.drop(['Win', 'Team', 'Player', 'Role'], axis=1)
y1 = new_data.Win


print("train X Data shape: {}".format(X_train.shape))
print("val X Data shape: {}".format(X_val.shape))
print("test X Data shape: {}".format(X_test.shape))



