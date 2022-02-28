[TOC]

### 1. 神经网络概述

首先需要搭建神经网络，喂入数据，也就是俗称进行训练，然后通过函数进行前向传播，通过得出的y以计算损失函数。

> 损失函数：预测值（y）与标准答案（y_）之间的差距。
>
> 损失函数可以定量判断在神经网络中的优劣，当损失函数最小的时候，神经网络中的参数W，b会出现最优值。
>
> 均方误差：$MSE(y - y_) = \sum_{k=0}^{n}{(y-y_ )^2}/n$

梯度下降

> 目的：想找到一组函数W，使得损失函数最小。
>
> 梯度：函数对各参数进行求偏导之后的向量。
>
> 梯度下降法：沿损失函数梯度下降的方法，寻找损失函数的最小值，，得到最优参数的方法。
>
> 反向传播：从后向前，逐层求损失函数对每层神经元参数的偏导数，迭代更新所有参数。

张量生成

`t = [[1,2,3][4,5,6]]     //二维张量`

### 2. tensorflow数据类型及常用函数

#### 2.1 数据类型

```python
tf.int   //tf.int 32
tf.float  //tf.float 32
tf.bool //tf.constant([True,False])
tf.string //tf.constant("hello world!")
```

#### 2.2 常用函数

##### 2.2.1创建一个张量：

`tf.constant(张量内容，dtype = 数据内容)`

```python
import tensorflow as tf
 
a = tf.constant([1, 5], dtype=tf.int64)
print("a:", a)
print("a.dtype:", a.dtype)
print("a.shape:", a.shape)

/*输出结果
a: tf.Tensor([1 5], shape=(2,), dtype=int64)
a.dtype: <dtype: 'int64'>
a.shape: (2,)
*/
```

##### 2.2.2将数据类型改为Tensor数据类型

很多时候数据类型是numpy格式的，可以使用 `tf.convert_to_tensor(数据名，dtype = 数据类型)`将数据类型改为Tensor数据类型。

```python
import tensorflow as tf
import numpy as np
 
a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print("a:", a)
print("b:", b)

/*输出结果：
a: [0 1 2 3 4]
b: tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)
*/
```

##### 2.2.3创建全为0的张量：

`tf.zeros(维度)`

##### 2.2.4创建全为1的张量

`tf.ones(维度)`

##### 2.2.5创建全为指定值的张量

`tf.fill(维度，指定值)`

```python
import tensorflow as tf
 
a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print("a:", a)
print("b:", b)
print("c:", c)

/*输出结果：
a: tf.Tensor([[0. 0. 0.][0. 0. 0.]], shape=(2, 3), dtype=float32)
b: tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
c: tf.Tensor([[9 9][9 9]], shape=(2, 2), dtype=int32)
*/
```

##### 2.2.6生成正态分布的随机数，默认均值为0，标准差为

`tf.random.normal(维度，mean=均值，stddev=标准差)`

##### 2.2.7生成截断式正态分布的随机数(数值更向均值集中)

`tf.random.truncated_notmal(维度，mean=均值，stddev=标准差)`

```python
import tensorflow as tf
 
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)

/*
d: tf.Tensor([[0.0576812  0.32228583] [2.1146383  0.4900563 ]], shape=(2, 2), dtype=float32)
e: tf.Tensor([[-0.49768662  1.8269136 ] [ 0.600802    1.0269477 ]], shape=(2, 2), dtype=float32)
*/
```

##### 2.2.8 生成均匀随机分布值（在区间内的均匀分布）

`tf.random.uniform(维度，minval = 最小值，maxval = 最大值)`

```python
import tensorflow as tf
 
f = tf.random.uniform([2, 2], minval=0, maxval=1)
print("f:", f)
/*
f: tf.Tensor([[0.577104  0.3696413]
 [0.9001155 0.6064184]], shape=(2, 2), dtype=float32)
*/
```

##### 2.2.9 强制tensor转换为该类型

`tf.cast(张量名，dtype=数据类型)`

##### 2.2.10 计算张量维度上元素的最小值

`tf.reduce_min(张量名)`

##### 2.2.11 计算张量维度上元素的最大值

`tf.reduce_max(张量名)`

```python
import tensorflow as tf
 
x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print("x1:", x1)
x2 = tf.cast(x1, tf.int32)
print("x2", x2)
print("minimum of x2：", tf.reduce_min(x2))
print("maxmum of x2:", tf.reduce_max(x2))

/*
x1: tf.Tensor([1. 2. 3.], shape=(3,), dtype=float64)
x2 tf.Tensor([1 2 3], shape=(3,), dtype=int32)
minimum of x2: tf.Tensor(1, shape=(), dtype=int32)
maxmum of x2: tf.Tensor(3, shape=(), dtype=int32)
*/
```

##### 2.2.12 理解axis

axis = 0表示跨行，axis = 1表示跨列

##### 2.2.13 计算沿着指定维度的平均值

`tf.reduce_mean(张量名，axis = 操作轴)`

##### 2.2.14 计算沿着指定维度的和

`tf.reduce_sum(张量名，axis = 操作轴)`

```python
import tensorflow as tf
 
x = tf.constant([[1, 2, 3], [2, 2, 3]])
print("x:", x)
print("mean of x:", tf.reduce_mean(x))  # 求x中所有数的均值
print("sum of x:", tf.reduce_sum(x, axis=1))  # 求每一行的和

/*
x: tf.Tensor([[1 2 3][2 2 3]], shape=(2, 3), dtype=int32)
mean of x: tf.Tensor(2, shape=(), dtype=int32)
sum of x: tf.Tensor([6 7], shape=(2,), dtype=int32)
*/
```

##### 2.2.15 tf.Variable()将变量标记为可训练

被标记的变量会在反向传播中记录梯度信息。

`tf.Variable(初始值)`

##### 2.2.16 常用的数学函数

张量a，b

对应元素的四则运算：`tf.add(a,b),tf.subtract(a,b),tf.multiply(a,b),tf.divide(a,b)`

平方，次方和开方：`tf.square(a,b),tf.pow(a,b),tf.sqrt`

矩阵乘：`tf.matmul`

```python
import tensorflow as tf
 
a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)
print("a:", a)
print("b:", b)
print("a+b:", tf.add(a, b))
print("a-b:", tf.subtract(a, b))
print("a*b:", tf.multiply(a, b))
print("b/a:", tf.divide(b, a))
/*
a: tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
b: tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
a+b: tf.Tensor([[4. 4. 4.]], shape=(1, 3), dtype=float32)
a-b: tf.Tensor([[-2. -2. -2.]], shape=(1, 3), dtype=float32)
a*b: tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
b/a: tf.Tensor([[3. 3. 3.]], shape=(1, 3), dtype=float32)
*/
```

```python
import tensorflow as tf
 
a = tf.ones([3, 2])
b = tf.fill([2, 3], 3.)
print("a:", a)
print("b:", b)
print("a*b:", tf.matmul(a, b))

/*
a: tf.Tensor(
[[1. 1.]
 [1. 1.]
 [1. 1.]], shape=(3, 2), dtype=float32)
b: tf.Tensor(
[[3. 3. 3.]
 [3. 3. 3.]], shape=(2, 3), dtype=float32)
a*b: tf.Tensor(
[[6. 6. 6.]
 [6. 6. 6.]
 [6. 6. 6.]], shape=(3, 3), dtype=float32)
*/
```

##### 2.2.17 特征标签配对函数，构建数据集

`data = tf.data.DataSet.from_tensor_slices(输入特征，标签)`

```python
import tensorflow as tf
 
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
for element in dataset:
    print(element)
    
/*
(<tf.Tensor: shape=(), dtype=int32, numpy=12>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
(<tf.Tensor: shape=(), dtype=int32, numpy=23>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
(<tf.Tensor: shape=(), dtype=int32, numpy=10>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
(<tf.Tensor: shape=(), dtype=int32, numpy=17>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
*/
```

##### 2.2.18 使用某个函数对指定参数的求导运算

```python
import tensorflow as tf

##with结构记录计算过程，gradient求出张量的梯度
with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))  ##若干个计算过程
    y = tf.pow(x, 2) 
grad = tape.gradient(y, x)  ##gradient（函数，对谁求导）
print(grad)

/*
tf.Tensor(6.0, shape=(), dtype=float32)
*/
```

##### 2.2.19 enumerate函数

它可以遍历每个元素（如列表，元组或字符串），组合为==索引，元素==，常在for循环中使用。

```python
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)

/*
0 one
1 two
2 three
*/
```

##### 2.2.20 独热编码，用作标签

标记类别：1表示是，0表示非

`tf.one_hot(待转换数据，depth = 几分类)`

```python
import tensorflow as tf
 
classes = 3
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels, depth=classes)
print("result of labels1:", output)
print("\n")

/*
result of labels1: tf.Tensor(
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]], shape=(3, 3), dtype=float32)
*/
```

##### 2.2.21 使输出符合规律分布

`tf.nn.softmax(x)`

```python
import tensorflow as tf
 
y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
 
print("After softmax, y_pro is:", y_pro)  # y_pro 符合概率分布
 
print("The sum of y_pro:", tf.reduce_sum(y_pro))  # 通过softmax后，所有概率加起来和为1

/*
After softmax, y_pro is: tf.Tensor([0.25598174 0.69583046 0.04818781], shape=(3,), dtype=float32)
The sum of y_pro: tf.Tensor(1.0, shape=(), dtype=float32)
*/
```

##### 2.2.22 赋值操作，更新参数的值并返回

`w.assign_sub(w要自减的内容)`

调用assign_sub之前，先用tf.Variable()定义变量w为可训练。

```python
import tensorflow as tf
 
x = tf.Variable(4)
x.assign_sub(1)
print("x:", x)  # 4-1=3
/*
x: <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>
*/
```

##### 2.2.23 返回张量沿指定维度最大值的索引

`tf.argmax(张量名，axis = 操作轴)`

```python
import numpy as np
import tensorflow as tf
 
test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print("test:\n", test)
print("每一列的最大值的索引：", tf.argmax(test, axis=0))  # 返回每一列最大值的索引
print("每一行的最大值的索引", tf.argmax(test, axis=1))  # 返回每一行最大值的索引

/*
每一列的最大值的索引: tf.Tensor([3 3 1], shape=(3,), dtype=int64)
每一列的最大值的索引 tf.Tensor([2 2 0 0], shape=(4,), dtype=int64)
*/
```

#### 2.3 读入数据集

```python
from sklearn import datasets
 
x_data = datasets.load_iris().data  # .data返回iris数据集所有输入特征
y_data = datasets.load_iris().target  # .target返回iris数据集所有标签
```

#### 2.4 神经网络实现步骤

##### 2.4.1 准备数据

- 数据集读入
- 数据集乱序
- 生成训练集和测试集（永不相交，一般比例为8：2）
- 配成（输入特征，标签）对，每次读入一小撮（batch）

##### 2.4.2 搭建网络

- 定义神经网络所有可训练参数

##### 2.4.3 参数优化

- 嵌套循环迭代，with结构更新参数，显示当前loss

##### 2.4.4 测试效果

- 计算当前参数向后传播的准确率，显示当前acc

##### 2.4.5 acc/loss可视化

#### 2.5 总体代码

```python
# -*- coding: UTF-8 -*-
# 利用鸢尾花数据集，实现前向传播、反向传播，可视化loss曲线
 
# 导入所需模块
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
 
# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
 
# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)
 
# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]
 
# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
 
# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
 
# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))
 
lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和
 
# 训练部分
for epoch in range(epoch):  #数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  #batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])
 
        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新
 
    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备
 
    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")
 
# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像
 
# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()
```

### 3.深入tensorflow

#### 3.1 预备知识

##### 3.1.1 tf.where()条件语句

`tf.where(条件语句，真返回A，否返回B)`

```python
import tensorflow as tf
 
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)  # 若a>b，返回a对应位置的元素，否则返回b对应位置的元素
print("c：", c)

/*
c： tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
*/
```

##### 3.1.2 返回一个【0，1】之间的随机数

`tf.random.RandomState.rand(维度)`  维度为空，返回标量

```python
import numpy as np
 
rdm = np.random.RandomState(seed=1)  #seed = 常数 每次生成的随机数相等
a = rdm.rand()   #返回一个随机标量
b = rdm.rand(2, 3)  #返回维度为两行三列随机数矩阵
print("a:", a)
print("b:", b)

/*
a: 0.417022004702574
b: [[7.20324493e-01 1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02 1.86260211e-01]]
*/
```

##### 3.1.3 两个数组按垂直方向叠加

`np.vstack(数组1，数组2)`

```python
import numpy as np
 
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
print("c:\n", c)

/*
c:
 [[1 2 3]
 [4 5 6]]
*/
```

##### 3.1.4 生成网格坐标点

`np.mgrid[起始值：结束值：步长，起始值：结束值：步长。。。。]`

`x.ravel()  将x变为一维数组，把“.”前面的变量拉直`

`np.c_[数组1，数组2，。。。]   使返回的间隔数值点配对`

```python
import numpy as np
import tensorflow as tf
 
# 生成等间隔数值点
x, y = np.mgrid[1:3:1, 2:4:0.5]
# 将x, y拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[x.ravel(), y.ravel()]
print("x:\n", x)
print("y:\n", y)
print("x.ravel():\n", x.ravel())
print("y.ravel():\n", y.ravel())
print('grid:\n', grid)

/*
x:
 [[1. 1. 1. 1.]
 [2. 2. 2. 2.]]
y:
 [[2.  2.5 3.  3.5]
 [2.  2.5 3.  3.5]]
x.ravel():
 [1. 1. 1. 1. 2. 2. 2. 2.]
y.ravel():
 [2.  2.5 3.  3.5 2.  2.5 3.  3.5]
grid:
 [[1.  2. ]
 [1.  2.5]
 [1.  3. ]
 [1.  3.5]
 [2.  2. ]
 [2.  2.5]
 [2.  3. ]
 [2.  3.5]]
*/
```

#### 3.2 神经网络（NN）复杂度

##### 3.2.1 空间复杂度

层数 = 隐藏层的层数+1个输出层

总参数=总w+总b

##### 3.2.2 时间复杂度

乘加运算次数

