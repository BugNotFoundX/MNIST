# MNIST

仅使用numpy手动构造的手写数字识别

## Usage

1. 安装`Python>=3.10`
2. 执行以下命令
   ```bash
   git clone https://github.com/FvTao/MNIST.git
   cd MNIST
   pip3 install numpy matplotlib
   python3 main.py
   ```

## BPNN 原理

### 概况

- BP(back propagation)神经网络是1986年由Rumelhart和McClelland为首的科学家提出的概念，是一种按照误差逆向传播算法训练的多层前馈神经网络，是应用最广泛的神经网络模型之一 。Minsky和Papert在颇具影响力的"perceptron" 一书中指出，简单的感知器只能求解线性问题，能够求解非线性问题的网络应该具有感知层，但是对隐藏层神经元的学习规则还没有合理的理论依据。

### 相关定义与原理

- 基本原理

  利用输出后的误差来估计输出层直接前层的误差，再利用这个误差估计更前一层的误差，如此递归

- 结构示意图![](assets/bpnn.bmp)

- 符号定义及相关运算

  假设输入层有n个神经元，隐含层有p个神经元，输出层含有q个神经元

  - 输入向量 $x = (x_1, x_2, x_3,...x_n)$

  - 隐含层输入向量 $hi = (hi_1, hi_2, hi_3,...hi_p)$

  - 隐含层输出向量 $ho = (ho_1, ho_2, ho_3,...ho_p)$

  - 输出层输入向量 $yi = (yi_1, yi_2, yi_3,...yi_q)$ 

  - 输出层输出向量 $yo = (yo_1, yo_2, yo_3,...yo_q)$

  - 输入层与中间层的连接权值 $w_{ih}$

  - 隐含层与输出层的连接权值 $w_{ho}$

  - 隐含层各神经元的阈值 $b_h$

  - 输出层各神经元的阈值 $b_o$

  - 样本数据个数 $k=1,2,3,...m$

  - 期望输出向量 $d = (d_1, d_2,d_3,...d_q)$

  - 误差函数 $e=\frac{1}{2} \sum_{i=0}^{q} (d_i(k) - yo_i(k))^2$ （第k个输入样本）

  - 激活函数 $f(x)$

  - 动量因子：BP神经网络在批处理训练时会陷入局部最小，这个时候加入一个动量因子有助于其反馈的误差信号使神经元的权值重新振荡起来

  - 偏置结点：偏置结点是为了描述训练数据中没有的特征，偏置结点对于下一层的每一个结点的权重的不同而生产不同的偏置，于是可以认为偏置是每一个结点（除输入层外）的属性
  
### 推导过程

- 随机选择第k个输入样本和其对应的期望输出

$$
\begin{gathered}
x(k) = \{x_1(k), x_2(k), x_3(k),...x_n(k)\}\\
d(k) = \{d_1(k), d_2(k), d_3(k),...d_q(k)\}
\end{gathered}
$$

- 计算隐含层的各个神经元的输入和输出

$$
\begin{aligned}
&h i_h(k)=\sum_{i=1}^n w_{i h} x_i(k)-b_h \quad h=1,2, \cdots, p\\
&h o_h(k)=f\left(h i_h(k)\right) \quad h=1,2, \cdots, p\\
&y i_o(k)=\sum_{h=1}^p w_{h o} h o_h(k)-b_o \quad o=1,2, \cdots q\\
&y o_o(k)=f\left(yi_o(k)\right) \quad o=1,2, \cdots, q
\end{aligned}
$$

- 利用网络期望输出和实际输出，计算误差函数e对输出层的各神经元的偏导数

$$
\begin{gathered}
\frac{\partial e}{\partial w_{h o}}=\frac{\partial e}{\partial y i_o} \frac{\partial y i_o}{\partial w_{h o}} \quad \\
\frac{\partial y i_o(k)}{\partial w_{h o}}=\frac{\partial\left(\sum_h^p w_{h o} h o_h(k)-b_o\right)}{\partial w_{h o}}=h o_h(k) \\
\frac{\partial e}{\partial y i_o}=\frac{\partial\left(\frac{1}{2} \sum_{o=1}^q\left(d_o(k)-y o_o(k)\right)\right)^2}{\partial y i_o}=-\left(d_o(k)-y o_o(k)\right) y o_o^{\prime}(k) \\
= -\left(d_o(k)-y o_o(k)\right) f'(yi_o(k)) = -\delta_o(k)
\end{gathered}
$$

- 利用输出层各神经元的 $\delta_o(k)$ 和隐含层各神经元的输出来修正连接权值 $w_{ho}(k)$ ， $\mu$ 是设置的学习率
  
$$
\begin{gathered}
\Delta w_{ho}(k) = -\mu \frac{\partial e}{\partial w_{ho}} = \mu\delta_0(k)ho_h(k) \\
w^{N+1}_{ho} = w^{N}_{ho} + \Delta w_{ho}(k)
\end{gathered}
$$

- 利用隐含层各神经元的 $\delta_o(k)$ 和输入层的输入来修正输出层的连接权 $w_{ho}(k)$

$$
\begin{gathered}
\Delta w_{i h}(k)
=-\mu \frac{\partial e}{\partial w_{i h}}
=-\mu \frac{\partial e}{\partial yi_o}\frac{\partial yi_o}{\partial ho_h(k)}\frac{\partial ho_h(k)}{\partial hi_h(k)}\frac{\partial h i_h(k)}{\partial w_{i h}}\\
\frac{\partial e}{\partial yi_o}= \delta_o(k),\quad 
\frac{\partial yi_o}{\partial ho_h(k)} = w_{ho},\quad
\frac{\partial ho_h(k)}{\partial hi_h(k)} = f'(hi_h(k))\\
\text{令}\ \delta_h(k) = \delta_o(k)w_{ho}f'(hi_h(k))\\
\text{所以}\ \Delta w_{i h}(k)=\mu\delta_h(k) x_i(k) \\
w_{i h}^{N+1}=w_{i h}^N+\mu \delta_h(k) x_i(k)
\end{gathered}
$$

- p.s.：求导时需要利用链式求导法则
  
- 全局误差

$$
E = \frac{1}{2m}\sum_{k=1}^{m}\sum_{i=1}^{n}(d_i(k) - yo_i(k))^2
$$

### 代码模板

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derived_sigmoid(x):
    return x * (1 - x)


def make_matrix(m, n, fill=0.0):  # m行n列
    a = []
    for i in range(m):
        a.append([fill] * n)
    return np.array(a)


# neural network class 4 layer
class neuralNetwork:

    # initialize the neural network
    def __init__(self, input_nodes, hidden_nodes, hidden2_nodes, output_nodes, learning_rate=0.1):
        """
        The network consists of three layers: input layer, hidden layer and output layer.
        Here defined these layers.
        :param input_nodes: dimension of input
        :param hidden_nodes: dimension of hidden nodes
        :param output_nodes: dimension of output
        :param learning_rate: the learning rate of neural network
        """

        self.x_num = input_nodes + 1  # 添加一个偏置
        self.h_num = hidden_nodes
        self.h2_num = hidden2_nodes
        self.y_num = output_nodes

        # 初始化向量
        self.x_vector = np.array([0.0] * self.x_num)
        self.hi_vector = np.array([0.0] * self.h_num)
        self.ho_vector = np.array([0.0] * self.h_num)
        self.h2i_vector = np.array([0.0] * self.h2_num)
        self.h2o_vector = np.array([0.0] * self.h2_num)
        self.yi_vector = np.array([0.0] * self.y_num)
        self.yo_vector = np.array([0.0] * self.y_num)

        # 初始化权值矩阵
        # 输入数据到隐藏层输入的变换矩阵
        self.weight_xh = (np.random.random([self.x_num, self.h_num]) - 0.51)  
        # 隐藏层间的变换矩阵
        self.weight_hh2 = (np.random.random([self.h_num, self.h2_num]) - 0.51)
        # 隐藏层输出到输出层输入的变换矩阵
        self.weight_h2y = (np.random.random([self.h2_num, self.y_num]) - 0.51)
        # 学习率
        self.lr = learning_rate

        # 动量因子
        self.input_correction = make_matrix(self.x_num, self.h_num)
        self.hidden_correction = make_matrix(self.h_num, self.h2_num)
        self.output_correction = make_matrix(self.h2_num, self.y_num)
		# 动量因子影响率
        self.m = 0.1

    def forward(self, input_feature):
        """
        Forward the neural network
        :param input_feature: single input image, flattened [784, ]
        """
        if len(input_feature) != self.x_num - 1:
            raise ValueError("输入数据与输入结点数量不同")

            # 简单的处理一下输入数据
        self.x_vector[1:self.x_num] = input_feature
        self.x_vector = np.array(self.x_vector)

        # 输入层->隐藏层
        self.hi_vector = np.dot(self.x_vector, self.weight_xh)
        # print(self.hi_vector)

        # 激活隐藏层神经元
        self.ho_vector = np.array(sigmoid(self.hi_vector))
        # print(self.ho_vector)

        self.h2i_vector = np.dot(self.ho_vector, self.weight_hh2)

        self.h2o_vector = np.array(sigmoid(self.h2i_vector))

        # 隐藏层->输出层
        self.yi_vector = np.dot(self.h2o_vector, self.weight_h2y)
        # print(self.yi_vector)

        # 激活输出层神经元
        self.yo_vector = np.array(sigmoid(self.yi_vector))
        # print(self.yo_vector)

        return self.yo_vector

    def backpropagation(self, targets_list):
        """
        Propagate backwards
        :param targets_list: output onehot code of a single image, [10, ]
        """
        if len(targets_list) != self.y_num:
            raise ValueError("标记数量与输出数量不符")

        targets = np.array(targets_list)  # 简单处理输入

        # 计算误差
        error = 0.5 * np.dot((targets - self.yo_vector).T,
                             (targets - self.yo_vector))

        # 计算残差
        delta_h2y = np.array((targets - self.yo_vector) 
                             * derived_sigmoid(self.yo_vector))
        delta_hh2 = np.array(np.dot(delta_h2y, self.weight_h2y.T) 
                             * derived_sigmoid(self.h2o_vector))
        delta_xh = np.array(np.dot(delta_hh2, self.weight_hh2.T) 
                            * derived_sigmoid(self.ho_vector))

        # 更新权值
        # print(self.weight_xh)
        self.weight_h2y += self.lr 
        	* np.dot(delta_h2y.reshape(-1, 1),self.h2o_vector.reshape(1, -1)).T 
        	+ self.m * self.output_correction

        self.weight_hh2 += self.lr 
        	* np.dot(delta_hh2.reshape(-1, 1),self.hi_vector.reshape(1, -1)).T 
        	+ self.m * self.hidden_correction

        self.weight_xh += self.lr 
        	* np.dot(delta_xh.reshape(-1, 1),self.x_vector.reshape(1, -1)).T 
        	+ self.m * self.input_correction

        # 更新
        self.output_correction = self.lr 
        	* np.dot(delta_h2y.reshape(-1, 1), self.h2o_vector.reshape(1, -1)).T
        self.hidden_correction = self.lr 
        	* np.dot(delta_hh2.reshape(-1, 1), self.ho_vector.reshape(1, -1)).T
        self.input_correction = self.lr 
        	* np.dot(delta_xh.reshape(-1, 1), self.x_vector.reshape(1, -1)).T
        return error
```



