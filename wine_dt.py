from sklearn.datasets import load_wine  # 红酒数据集
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # 引入决策树分类模型
# 可视化决策树
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = load_wine(as_frame=True)  # 导入红酒数据集


X = data.data  # 特征矩阵
Y = data.target  # 标签

# 划分训练集与测试集
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2)  # test_size为测试集所占比例，此处为20%

DTC = DecisionTreeClassifier(criterion="entropy")  # 实例化决策树分类模型
DTC = DTC.fit(X_train, Y_train)  # 应用训练集进行分类学习


# plt.figure(figsize=(15,20))  # 新建画布，figsize设置画布大小
plot_tree(DTC, feature_names=X.columns, filled=True)  # 绘制决策树；filled=True表示颜色填充，颜色越深纯度越高；feature_names为特征名称
plt.show()

# 查看决策树模型在测试集上的表现
score_test = DTC.score(X_test, Y_test)  # 返回测试集上的分类准确率
print(score_test)

Y_predict = DTC.predict(X_test)  # 查看该模型对于测试集的分类预测结果
print(Y_predict)
