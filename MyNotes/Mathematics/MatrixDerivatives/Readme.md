# 矩阵的导数运算学习笔记

## 1 标量函数对向量求导

### 1.1 基本定义

设标量函数 $f(\mathbf{x})$，其中 $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ 是 $n$ 维列向量。

### 1.2 分子形式 (Numerator Layout)

分子形式将求导变量视为分子，结果的形状与求导变量相同。

#### 1.2.1 定义
$$\frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

#### 1.2.2 特点
- 结果为 $n \times 1$ 的列向量
- 梯度向量的标准形式
- 常用于机器学习和优化理论

### 1.3 分母形式 (Denominator Layout)

分母形式将求导变量视为分母，结果的形状与求导变量的转置相同。

#### 1.3.1 定义
$$\frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}$$

#### 1.3.2 特点
- 结果为 $1 \times n$ 的行向量
- 有时在统计学中使用
- 与分子形式互为转置

### 1.4 常用例子

#### 1.4.1 线性函数
设 $f(\mathbf{x}) = \mathbf{a}^T \mathbf{x}$，其中 $\mathbf{a}$ 是常向量。

**分子形式：**
$$\frac{\partial f}{\partial \mathbf{x}} = \mathbf{a}$$

**分母形式：**
$$\frac{\partial f}{\partial \mathbf{x}} = \mathbf{a}^T$$

#### 1.4.2 二次函数
设 $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$，其中 $\mathbf{A}$ 是 $n \times n$ 矩阵。

**分子形式：**
$$\frac{\partial f}{\partial \mathbf{x}} = (\mathbf{A} + \mathbf{A}^T) \mathbf{x}$$

**分母形式：**
$$\frac{\partial f}{\partial \mathbf{x}} = \mathbf{x}^T (\mathbf{A} + \mathbf{A}^T)$$

### 1.5 选择建议

- **推荐使用分子形式**：与标准梯度定义一致，在机器学习和深度学习中更常见
- **保持一致性**：在同一个项目中应始终使用同一种形式
- **注意转换**：两种形式之间只相差一个转置操作

### 1.6 链式法则

在分子形式下，链式法则为：
$$\frac{\partial f}{\partial \mathbf{x}} = \frac{\partial f}{\partial \mathbf{u}} \frac{\partial \mathbf{u}}{\partial \mathbf{x}}$$

其中 $\mathbf{u} = \mathbf{u}(\mathbf{x})$ 是中间变量。

## 2 向量函数对向量求导

### 2.1 基本定义

设(列)向量函数 $\mathbf{f}(\mathbf{x}) = [f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_m(\mathbf{x})]^T$，其中 $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ 是 $n$ 维列向量。

### 2.2 分子形式 (Numerator Layout)

分子形式下，Jacobian矩阵的定义为：

#### 2.2.1 定义
$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

#### 2.2.2 特点
- 结果为 $m \times n$ 矩阵（Jacobian矩阵）
- 第 $i$ 行是 $f_i$ 对 $\mathbf{x}$ 的梯度转置
- 常用于多元函数的线性化

### 2.3 分母形式 (Denominator Layout)

分母形式下，结果是分子形式的转置：

#### 2.3.1 定义
$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_2}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_1} \\
\frac{\partial f_1}{\partial x_2} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_2} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_1}{\partial x_n} & \frac{\partial f_2}{\partial x_n} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

#### 2.3.2 特点
- 结果为 $n \times m$ 矩阵
- 第 $j$ 列是 $f_j$ 对 $\mathbf{x}$ 的梯度
- 有时在某些统计应用中使用

### 2.4 常用例子

#### 2.4.1 线性变换
设 $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x} + \mathbf{b}$，其中 $\mathbf{A}$ 是 $m \times n$ 矩阵，$\mathbf{b}$ 是 $m$ 维向量。

**分子形式：**
$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \mathbf{A}$$

**分母形式：**
$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \mathbf{A}^T$$

#### 2.4.2 二次形式
设 $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x}$ 且 $f_i(\mathbf{x}) = \mathbf{a}_i^T \mathbf{x}$，其中 $\mathbf{a}_i$ 是 $\mathbf{A}$ 的第 $i$ 行。

**分子形式：**
$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \mathbf{A}$$

### 2.5 链式法则

对于复合向量函数 $\mathbf{f}(\mathbf{g}(\mathbf{x}))$：

**分子形式：**
$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \frac{\partial \mathbf{f}}{\partial \mathbf{g}} \frac{\partial \mathbf{g}}{\partial \mathbf{x}}$$

其中：
- $\frac{\partial \mathbf{f}}{\partial \mathbf{g}}$ 是 $m \times p$ 矩阵
- $\frac{\partial \mathbf{g}}{\partial \mathbf{x}}$ 是 $p \times n$ 矩阵
- 结果是 $m \times n$ 矩阵

### 2.6 特殊情况

#### 2.6.1 向量对自身求导
$$\frac{\partial \mathbf{x}}{\partial \mathbf{x}} = \mathbf{I}_n$$

其中 $\mathbf{I}_n$ 是 $n \times n$ 的单位矩阵。

#### 2.6.2 矩阵向量乘积的求导
这是一个非常重要的特殊性质，在机器学习中经常用到。

设 $\mathbf{f}(\mathbf{y}) = \mathbf{A}\mathbf{y}$，其中 $\mathbf{A}$ 是 $m \times n$ 常数矩阵，$\mathbf{y}$ 是 $n$ 维向量。

**分子形式：**
$$\frac{\partial (\mathbf{A}\mathbf{y})}{\partial \mathbf{y}} = \mathbf{A}$$

**分母形式：**
$$\frac{\partial (\mathbf{A}\mathbf{y})}{\partial \mathbf{y}} = \mathbf{A}^T$$

**推导说明：**
- $\mathbf{A}\mathbf{y} = \begin{bmatrix} \mathbf{a}_1^T\mathbf{y} \\ \mathbf{a}_2^T\mathbf{y} \\ \vdots \\ \mathbf{a}_m^T\mathbf{y} \end{bmatrix}$，其中 $\mathbf{a}_i^T$ 是 $\mathbf{A}$ 的第 $i$ 行
- 第 $i$ 个分量 $f_i = \mathbf{a}_i^T\mathbf{y} = a_{i1}y_1 + a_{i2}y_2 + \cdots + a_{in}y_n$
- 因此 $\frac{\partial f_i}{\partial y_j} = a_{ij}$
- 在分子形式下，Jacobian矩阵的第 $(i,j)$ 元素是 $\frac{\partial f_i}{\partial y_j} = a_{ij}$，所以结果是 $\mathbf{A}$

**实际应用：**
- **线性层前向传播**：$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$，则 $\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \mathbf{W}$
- **反向传播**：如果损失对 $\mathbf{z}$ 的梯度是 $\frac{\partial L}{\partial \mathbf{z}}$，则对 $\mathbf{x}$ 的梯度是 $\mathbf{W}^T \frac{\partial L}{\partial \mathbf{z}}$

#### 2.6.3 二次型标量函数的求导

设标量函数 $f(\mathbf{y}) = \mathbf{y}^T \mathbf{A} \mathbf{y}$，其中 $\mathbf{A}$ 是 $n \times n$ 矩阵，$\mathbf{y}$ 是 $n$ 维列向量。

**分子形式：**
$$\frac{\partial (\mathbf{y}^T \mathbf{A} \mathbf{y})}{\partial \mathbf{y}} = \mathbf{A}\mathbf{y} + \mathbf{A}^T\mathbf{y} = (\mathbf{A} + \mathbf{A}^T)\mathbf{y}$$

**分母形式：**
$$\frac{\partial (\mathbf{y}^T \mathbf{A} \mathbf{y})}{\partial \mathbf{y}} = \mathbf{y}^T(\mathbf{A} + \mathbf{A}^T)$$

**特殊情况（对称矩阵）：**
当 $\mathbf{A}$ 是对称矩阵（即 $\mathbf{A} = \mathbf{A}^T$）时：

分子形式：
$$\frac{\partial (\mathbf{y}^T \mathbf{A} \mathbf{y})}{\partial \mathbf{y}} = 2\mathbf{A}\mathbf{y}$$

分母形式：
$$\frac{\partial (\mathbf{y}^T \mathbf{A} \mathbf{y})}{\partial \mathbf{y}} = 2\mathbf{y}^T\mathbf{A}$$

**推导说明：**
展开二次型：
$$\mathbf{y}^T \mathbf{A} \mathbf{y} = \sum_{i=1}^n \sum_{j=1}^n a_{ij} y_i y_j$$

对 $y_k$ 求偏导：
$$\frac{\partial (\mathbf{y}^T \mathbf{A} \mathbf{y})}{\partial y_k} = \sum_{i=1}^n a_{ik} y_i + \sum_{j=1}^n a_{kj} y_j = (\mathbf{A}\mathbf{y})_k + (\mathbf{A}^T\mathbf{y})_k$$

因此梯度向量为：
$$\nabla_{\mathbf{y}} (\mathbf{y}^T \mathbf{A} \mathbf{y}) = \mathbf{A}\mathbf{y} + \mathbf{A}^T\mathbf{y}$$

**常见应用：**
- **线性代数**：计算协方差矩阵的梯度
- **机器学习**：正则化项 $\lambda \mathbf{w}^T \mathbf{w}$ 的梯度为 $2\lambda \mathbf{w}$
- **优化理论**：二次目标函数的梯度计算
- **最小二乘法**：损失函数 $(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ 的梯度

**具体例子：**
1. $f(\mathbf{y}) = \mathbf{y}^T \mathbf{y}$（欧几里得范数的平方）：
   $$\frac{\partial f}{\partial \mathbf{y}} = 2\mathbf{y}$$

2. $f(\mathbf{y}) = \mathbf{y}^T \mathbf{Q} \mathbf{y}$，其中 $\mathbf{Q}$ 是正定对称矩阵：
   $$\frac{\partial f}{\partial \mathbf{y}} = 2\mathbf{Q}\mathbf{y}$$

#### 2.6.4 梯度的特殊情况
当 $m = 1$ 时，向量函数退化为标量函数，Jacobian矩阵退化为梯度向量。

#### 2.6.5 行向量函数对列向量求导

对于 $1 \times m$ 行向量函数 $\mathbf{f}^T(\mathbf{x}) = [f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_m(\mathbf{x})]$ 对 $n \times 1$ 列向量 $\mathbf{x}$ 求导：

**分子形式：**
$$\frac{\partial \mathbf{f}^T}{\partial \mathbf{x}} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_2}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_1} \\
\frac{\partial f_1}{\partial x_2} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_2} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_1}{\partial x_n} & \frac{\partial f_2}{\partial x_n} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

**分母形式：**
$$\frac{\partial \mathbf{f}^T}{\partial \mathbf{x}} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

**特点说明：**
- 分子形式结果为 $n \times m$ 矩阵，第 $j$ 列是 $f_j$ 对 $\mathbf{x}$ 的梯度
- 分母形式结果为 $m \times n$ 矩阵，第 $i$ 行是 $f_i$ 对 $\mathbf{x}$ 的梯度转置
- **与列向量函数对向量求导的结果相比，两种形式对调**

**重要关系：**
如果 $\mathbf{g}(\mathbf{x}) = [\mathbf{f}^T(\mathbf{x})]^T$ 是对应的列向量函数，则：
$$\frac{\partial \mathbf{f}^T}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{g}}{\partial \mathbf{x}}\right)^T$$

**常见例子：**
设 $\mathbf{f}^T(\mathbf{x}) = \mathbf{x}^T \mathbf{A}$，其中 $\mathbf{A}$ 是 $n \times m$ 矩阵。

分子形式：
$$\frac{\partial (\mathbf{x}^T \mathbf{A})}{\partial \mathbf{x}} = \mathbf{A}$$

分母形式：
$$\frac{\partial (\mathbf{x}^T \mathbf{A})}{\partial \mathbf{x}} = \mathbf{A}^T$$

**应用场景：**
- **线性回归的梯度计算**：当预测值为 $\hat{\mathbf{y}}^T = \mathbf{x}^T \mathbf{W}$ 时
- **注意力机制**：计算注意力权重对输入的梯度
- **优化问题**：当目标函数输出为行向量时的梯度计算

### 2.7 向量函数对向量求导应用场景

- **神经网络反向传播**：计算损失函数对权重的导数
- **最优化算法**：Newton法中需要计算Hessian矩阵
- **数值分析**：函数的线性近似和敏感性分析
- **控制理论**：系统的线性化和稳定性分析

