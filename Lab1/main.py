import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from minepy import MINE
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
# from IPython.display import display
from datetime import datetime as dt
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.metrics import fbeta_score, accuracy_score, recall_score

warnings.filterwarnings('ignore')
# %matplotlib inline

def processing_data(data_path):
    """
    数据处理
    :param data_path: 数据集路径
    :return: feature1,feature2,label: 处理后的特征数据、标签数据
    """

    # 导入医疗数据
    data_xls = pd.ExcelFile(data_path)
    data = {}

    # 查看数据名称与大小
    for name in data_xls.sheet_names:
        df = data_xls.parse(sheet_name=name, header=None)
        data[name] = df

    # 获取 特征1 特征2 类标
    feature1_raw = data['Feature1']
    feature2_raw = data['Feature2']
    label = data['label']

    # 初始化一个 scaler，并将它施加到特征上
    scaler = MinMaxScaler()
    feature1 = pd.DataFrame(scaler.fit_transform(feature1_raw))
    feature2 = pd.DataFrame(scaler.fit_transform(feature2_raw))

    return feature1, feature2, label

def plot3D(feature,label):
    fig = plt.figure()
    if feature.shape[1] != 3:
        if feature.shape[1] ==1:
            ax = fig.add_subplot(121)
            ax.scatter(feature[:,0], np.random.rand(feature.shape[0]), c=label)
            ax2 = fig.add_subplot(122)
            ax2.scatter(feature[:,0], c=label)
        else:
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(feature[:,0], feature[:,1],np.random.rand(feature.shape[0]), c=label)
            ax = fig.add_subplot(122)
            ax.scatter(feature[:,0], feature[:,1], c=label)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(feature[:,0], feature[:,1], feature[:,2], c=label)
    plt.show()
def feature_select(feature1, feature2, label):
    """
    特征选择
    :param  feature1,feature2,label: 数据处理后的输入特征数据，标签数据
    :return: new_features,label:特征选择后的特征数据、标签数据
    """
    new_features = None
    # -------------------------- 实现特征选择部分代码 ----------------------------
    # 选择降维维度
    # tsne = TSNE(n_components=3)
    # feature_tsne = tsne.fit_transform(feature1)

    # select_feature_number1 = 66
    # select_feature_number2 = 30
    select_feature_number1 = 30
    select_feature_number2 = 15
    select_feature1 = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number1
                                  ).fit(feature1, np.array(label).flatten()).get_support(indices=True)

    select_feature2 = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number2
                                  ).fit(feature2, np.array(label).flatten()).get_support(indices=True)

    # feature1 = pd.concat([feature1[feature1.columns.values[select_feature1]],
    #                                feature2[feature2.columns.values[select_feature2]]], axis=1)

    feature1=feature1[feature1.columns.values[select_feature1]]
    feature2=feature2[feature2.columns.values[select_feature2]]
    # print(feature1.shape,"|",feature2.shape)

    # tsne = TSNE(n_components=3)
    # feature_tsne = tsne.fit_transform(feature1)

    # decomp = PCA(n_components=3)
    decomp1 = KernelPCA(n_components=2, kernel="sigmoid",gamma=0.3)
    feature_tsne1 = decomp1.fit_transform(feature1)


    decomp2 = KernelPCA(n_components=1, kernel="sigmoid",gamma=0.10)
    feature_tsne2 = decomp2.fit_transform(feature2)

    # tsne_label = np.array(label).flatten()
    # # print("new_features shape:", feature_tsne.shape)
    # plot3D(feature_tsne1, tsne_label)

    new_features = np.concatenate((feature_tsne1, feature_tsne2), axis=1)

    # decomp = KernelPCA(n_components=3, kernel="rbf", gamma=0.1)
    # feature_tsne = decomp.fit_transform(new_features)

    # 可视化类标中不能出现负值
    tsne_label = np.array(label).flatten()
    # print("new_features shape:", feature_tsne.shape)
    plot3D(new_features, tsne_label)
    # 双模态特征选择并融合

    print("new_features shape:", new_features.shape)
    # ------------------------------------------------------------------------
    # 返回筛选后的数据
    return new_features, label

def train_model(features, label):
    X_train, X_val, y_train, y_val = train_test_split(features, label, test_size=0.2, random_state=0, stratify=label)
    results = {}

    # 使用训练集数据来拟合学习器
    mylearner = naive_bayes.GaussianNB().fit(features, label)
    joblib.dump(mylearner, './results/my_model.m')

    learner=joblib.load('./results/my_model.m')
    # 得到在验证集上的预测值
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train)
    # 计算在训练数据的准确率
    results['acc_train'] = round(accuracy_score(y_train, predictions_train),4)

    # 计算在验证上的准确率
    results['acc_val'] = round(accuracy_score(y_val, predictions_val),4)

    # 计算在训练数据上的召回率
    results['recall_train'] = round(recall_score(y_train, predictions_train),4)

    # 计算验证集上的召回率
    results['recall_val'] = round(recall_score(y_val, predictions_val),4)

    # 计算在训练数据上的F-score
    results['f_train'] = round(fbeta_score(y_train, predictions_train, beta=1),4)

    # 计算验证集上的F-score
    results['f_val'] = round(fbeta_score(y_val, predictions_val, beta=1),4)

    print(results)
def data_split(features, label):
    """
    数据切分
    :param  features,label: 特征选择后的输入特征数据、类标数据
    :return: X_train, X_val, X_test,y_train, y_val, y_test:数据切分后的训练数据、验证数据、测试数据
    """

    # X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None
    # -------------------------- 实现数据切分部分代码 ----------------------------
    # 将 features 和 label 数据切分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0, stratify=label)

    # 将 X_train 和 y_train 进一步切分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

    # return X_train, X_val, X_test, y_train, y_val, y_test
    # ------------------------------------------------------------------------

    return X_train, X_val, X_test, y_train, y_val, y_test




def train_predict(learner, X_train, y_train, X_val, y_val):
    '''
    模型训练验证
    :param learner: 监督学习模型
    :param X_train: 训练集 特征数据
    :param y_train: 训练集 类标
    :param X_val: 验证集 特征数据
    :param y_val: 验证集 类标
    :return: results: 训练与验证结果
    '''

    results = {}

    # 使用训练集数据来拟合学习器
    start = time()  # 获得程序开始时间
    learner = learner.fit(X_train, y_train)
    end = time()  # 获得程序结束时间

    # 计算训练时间
    # results['train_time'] = end - start

    # 得到在验证集上的预测值
    start = time()  # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train)
    end = time()  # 获得程序结束时间

    # 计算预测用时
    # results['pred_time'] = end - start

    # 计算在训练数据的准确率
    results['acc_train'] = round(accuracy_score(y_train, predictions_train),4)

    # 计算在验证上的准确率
    results['acc_val'] = round(accuracy_score(y_val, predictions_val),4)

    # 计算在训练数据上的召回率
    results['recall_train'] = round(recall_score(y_train, predictions_train),4)

    # 计算验证集上的召回率
    results['recall_val'] = round(recall_score(y_val, predictions_val),4)

    # 计算在训练数据上的F-score
    results['f_train'] = round(fbeta_score(y_train, predictions_train, beta=1),4)

    # 计算验证集上的F-score
    results['f_val'] = round(fbeta_score(y_val, predictions_val, beta=1),4)

    # 成功
    print("{} trained on {} samples.".format(learner.__class__.__name__, len(X_val)))

    # 返回结果
    return results

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y, cv=None, n_jobs=1):
    """
    绘制学习曲线
    :param estimator: 训练好的模型
    :param X:绘制图像的 X 轴数据
    :param y:绘制图像的 y 轴数据
    :param cv: 交叉验证
    :param n_jobs:
    :return:
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure('Learning Curve', facecolor='lightgray')
    plt.title('Learning Curve')
    plt.xlabel('train size')
    plt.ylabel('score')
    plt.grid(linestyle=":")
    plt.plot(train_sizes, train_scores_mean, label='traning score')
    plt.plot(train_sizes, test_scores_mean, label='val score')
    plt.legend()
    plt.show()
def search_model(X_train, y_train, X_val, y_val, model_save_path):
    """
    创建、训练、优化和保存深度学习模型
    :param X_train, y_train: 训练集数据
    :param X_val,y_val: 验证集数据
    :param save_model_path: 保存模型的路径和名称
    :return:
    """
    # --------------------- 实现模型创建、训练、优化和保存等部分的代码 ---------------------
    clf_A = tree.DecisionTreeClassifier(random_state=42)
    clf_B = naive_bayes.GaussianNB()
    clf_C = svm.SVC()

    # 收集学习器的结果
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        results[clf_name] = train_predict(clf, X_train, y_train, X_val, y_val)
    print("高斯朴素贝叶斯模型结果:", results['GaussianNB'])
    print("支持向量机模型结果:", results['SVC'])
    print("决策树模型结果:", results['DecisionTreeClassifier'])
    # return

    #创建监督学习模型 以决策树为例
    clf = tree.DecisionTreeClassifier(random_state=42)

    # 创建调节的参数列表
    parameters = {'max_depth': range(1,10),
                  'min_samples_split': range(1,10)}

    # 创建一个fbeta_score打分对象 以F-score为例
    scorer = make_scorer(fbeta_score, beta=1)

    # 在分类器上使用网格搜索，使用'scorer'作为评价函数
    kfold = KFold(n_splits=10) #切割成十份

    # 同时传入交叉验证函数
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer, cv=kfold)

    #绘制学习曲线
    plot_learning_curve(clf, X_train, y_train, cv=kfold, n_jobs=4)

    # 用训练数据拟合网格搜索对象并找到最佳参数
    grid_obj.fit(X_train, y_train)

    # 得到estimator并保存
    best_clf = grid_obj.best_estimator_
    joblib.dump(best_clf, model_save_path)

    # 使用没有调优的模型做预测
    predictions = (clf.fit(X_train, y_train)).predict(X_val)
    best_predictions = best_clf.predict(X_val)

    # 调优后的模型
    print ("best_clf\n------")
    print (best_clf)

    # 汇报调参前和调参后的分数
    print("\nUnoptimized model\n------")
    print("Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions)))
    print("Recall score on validation data: {:.4f}".format(recall_score(y_val, predictions)))
    print("F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 1)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions)))
    print("Recall score on validation data: {:.4f}".format(recall_score(y_val, best_predictions)))
    print("Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 1)))

    # 保存模型（请写好保存模型的路径及名称）
    # -------------------------------------------------------------------------


def load_and_model_prediction(X_test, y_test, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型优化过程中的参数选择，测试集数据的准确率、召回率、F-score 等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param X_test,y_test: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------

    # ---------------------------------------------------------------------------


def main():
    """
    监督学习模型训练流程, 包含数据处理、特征选择、训练优化模型、模型保存、评价模型等。
    如果对训练出来的模型不满意, 你可以通过修改数据处理方法、特征选择方法、调整模型类型和参数等方法重新训练模型, 直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意, 则可以进行测试提交!
    :return:
    """
    data_path = "DataSet.xlsx"  # 数据集路径

    save_model_path = './results/my_model.m'  # 保存模型路径和名称

    # 获取数据 预处理
    feature1, feature2, label = processing_data(data_path)

    # 特征选择
    new_features, label = feature_select(feature1, feature2, label)
    # train_model(new_features, label)
    # return
    # 数据划分
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(new_features, label)

    # 创建、训练和保存模型
    search_model(X_train, y_train, X_val, y_val, save_model_path)
    #
    # # 评估模型
    # load_and_model_prediction(X_test, y_test, save_model_path)


if __name__ == '__main__':
    main()