# Why Numpy & Pandas?
# 应用
- 数据分析
- 机器学习
- 深度学习

## 为什么使用numpy & pandas
- 运算速度快：numpy 和 pandas 都是采用 C 语言编写, pandas 又是基于 numpy, 是 numpy 的升级版本。
- 消耗资源少：采用的是矩阵运算，会比 python 自带的字典或者列表快好多。

# Numpy 和 Pandas 安装
# numpy 安装
- sudo apt - get install python - numpy
- 使用 python 3 + : pip3 install numpy
- 使用 python 2 + : pip install numpy

# pandas 安装
- sudo apt - get install python - pandas
- 使用 python 3 + : pip3 install pandas
- 使用 python 2 + : pip install pandas

# Numpy 属性
这次我们会介绍几种 numpy 的属性:
- ndim: 维度
- shape: 行数和列数
- size: 元素个数

使用numpy首先要导入模块
```python
import numpy as np  # 为了方便使用numpy 采用np简写
```

列表转化为矩阵：
```python
array = np.array([[1, 2, 3], [2, 3, 4]])  # 列表转化为矩阵
print(array)
"""
array([[1, 2, 3],
       [2, 3, 4]])
"""
```

# numpy 的几种属性
接下来我们看看这几种属性的结果：

```python
print('number of dim:', array.ndim)  # 维度
# number of dim: 2

print('shape :', array.shape)    # 行数和列数
# shape : (2, 3)

print('size:', array.size)   # 元素个数
# size: 6
```

# Numpy 的创建 array
# 关键字
- array：创建数组
- dtype：指定数据类型
- zeros：创建数据全为0
- ones：创建数据全为1
- empty：创建数据接近0
- arrange：按指定范围创建数据
- linspace：创建线段

# 创建数组

```python
a = np.array([2, 23, 4])  # list 1d
print(a)
# [2 23 4]
```

# 指定数据 dtype

```python
a = np.array([2, 23, 4], dtype=np.int)
print(a.dtype)
# int 64
```

```python
a = np.array([2, 23, 4], dtype=np.int32)
print(a.dtype)
# int32
```

```python
a = np.array([2, 23, 4], dtype=np.float)
print(a.dtype)
# float64
```

```python
a = np.array([2, 23, 4], dtype=np.float32)
print(a.dtype)
# float32
```

# 创建特定数据

```python
a = np.array([[2, 23, 4], [2, 32, 4]])  # 2d 矩阵 2行3列
print(a)
"""
[[ 2 23  4]
 [ 2 32  4]]
"""
```

创建全零数组
```python
a = np.zeros((3, 4))  # 数据全为0，3行4列
"""
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
"""
```

创建全一数组, 同时也能指定这些特定数据的 dtype:
```python
a = np.ones((3, 4), dtype=np.int)   # 数据为1，3行4列
"""
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
"""
```

创建全空数组, 其实每个值都是接近于零的数:
```python
a = np.empty((3, 4))  # 数据为empty，3行4列
"""
array([[  0.00000000e+000,   4.94065646e-324,   9.88131292e-324,
          1.48219694e-323],
       [  1.97626258e-323,   2.47032823e-323,   2.96439388e-323,
          3.45845952e-323],
       [  3.95252517e-323,   4.44659081e-323,   4.94065646e-323,
          5.43472210e-323]])
"""
```

用 arange 创建连续数组:
```python
a = np.arange(10, 20, 2)  # 10-19 的数据，2步长
"""
array([10, 12, 14, 16, 18])
"""
```

使用 reshape 改变数据的形状
```python
a = np.arange(12).reshape((3, 4))    # 3行4列，0到11
"""
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
"""
```

用 linspace 创建线段型数据:
```python
a = np.linspace(1, 10, 20)    # 开始端1，结束端10，且分割成20个数据，生成线段
"""
array([  1.        ,   1.47368421,   1.94736842,   2.42105263,
         2.89473684,   3.36842105,   3.84210526,   4.31578947,
         4.78947368,   5.26315789,   5.73684211,   6.21052632,
         6.68421053,   7.15789474,   7.63157895,   8.10526316,
         8.57894737,   9.05263158,   9.52631579,  10.        ])
"""
```

同样也能进行 reshape 工作:
```python
a = np.linspace(1, 10, 20).reshape((5, 4))  # 更改shape
"""
array([[  1.        ,   1.47368421,   1.94736842,   2.42105263],
       [  2.89473684,   3.36842105,   3.84210526,   4.31578947],
       [  4.78947368,   5.26315789,   5.73684211,   6.21052632],
       [  6.68421053,   7.15789474,   7.63157895,   8.10526316],
       [  8.57894737,   9.05263158,   9.52631579,  10.        ]])
"""
```

# Numpy 基础运算1

让我们从一个脚本开始了解相应的计算以及表示形式 ：

```python
import numpy as np
a = np.array([10, 20, 30, 40])   # array([10, 20, 30, 40])
b = np.arange(4)              # array([0, 1, 2, 3])
```

# numpy 的几种基本运算

减法，加法，乘法：
```python
c = a - b  # array([10, 19, 28, 37])
c = a + b   # array([10, 21, 32, 43])
c = a * b   # array([  0,  20,  60, 120])
```

有所不同的是，在Numpy中，想要求出矩阵中各个元素的乘方需要依赖双星符号 **，以二次方举例，即：
```python
c = b**2  # array([0, 1, 4, 9])
```

另外，Numpy中具有很多的数学函数工具，比如三角函数等，当我们需要对矩阵中每一项元素进行函数运算时，可以很简便的调用它们（以sin函数为例）：
```python
c = 10 * np.sin(a)
# array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 ])
```

除了函数应用外，在脚本中对print函数进行一些修改可以进行逻辑判断：
```python
print(b < 3)
# array([ True,  True,  True, False], dtype=bool)
```

此时由于进行逻辑判断，返回的是一个bool类型的矩阵，即对满足要求的返回True，不满足的返回False。上述程序执行后得到的结果是[True True True False]。 需要注意的是，如果想要执行是否相等的判断， 依然需要输入 == 而不是 = 来完成相应的逻辑判断。

上述运算均是建立在一维矩阵，即只有一行的矩阵上面的计算，如果我们想要对多行多维度的矩阵进行操作，需要对开始的脚本进行一些修改：
```python
a = np.array([[1, 1], [0, 1]])
b = np.arange(4).reshape((2, 2))

print(a)
# array([[1, 1],
#       [0, 1]])

print(b)
# array([[0, 1],
#       [2, 3]])
```

此时构造出来的矩阵a和b便是2行2列的，其中 reshape 操作是对矩阵的形状进行重构， 其重构的形状便是括号中给出的数字。 稍显不同的是，Numpy中的矩阵乘法分为两种， 其一是前文中的对应元素相乘，其二是标准的矩阵乘法运算，即对应行乘对应列得到相应元素：
```python
c_dot = np.dot(a, b)
# array([[2, 4],
#       [2, 3]])
```

除此之外还有另外的一种关于dot的表示方法，即：
```python
c_dot_2 = a.dot(b)
# array([[2, 4],
#       [2, 3]])
```

下面我们将重新定义一个脚本, 来看看关于 sum(), min(), max()的使用：
```python
import numpy as np
a = np.random.random((2, 4))
print(a)
# array([[ 0.94692159,  0.20821798,  0.35339414,  0.2805278 ],
#       [ 0.04836775,  0.04023552,  0.44091941,  0.21665268]])
```

因为是随机生成数字, 所以你的结果可能会不一样. 在第二行中对a的操作是令a中生成一个2行4列的矩阵，且每一元素均是来自从0到1的随机数。 在这个随机生成的矩阵中，我们可以对元素进行求和以及寻找极值的操作，具体如下：
```python
np.sum(a)   # 4.4043622002745959
np.min(a)   # 0.23651223533671784
np.max(a)   # 0.90438450240606416
```

对应的便是对矩阵中所有元素进行求和，寻找最小值，寻找最大值的操作。 可以通过print()函数对相应值进行打印检验。
如果你需要对行或者列进行查找运算，就需要在上述代码中为 axis 进行赋值。 当axis的值为0的时候，将会以列作为查找单元， 当axis的值为1的时候，将会以行作为查找单元。
为了更加清晰，在刚才的例子中我们继续进行查找：
```python
print("a =", a)
# a = [[ 0.23651224  0.41900661  0.84869417  0.46456022]
# [ 0.60771087  0.9043845   0.36603285  0.55746074]]

print("sum =", np.sum(a, axis=1))
# sum = [ 1.96877324  2.43558896]

print("min =", np.min(a, axis=0))
# min = [ 0.23651224  0.41900661  0.36603285  0.46456022]

print("max =", np.max(a, axis=1))
# max = [ 0.84869417  0.9043845 ]
```

# Numpy 基础运算2

通过上一节的学习，我们可以了解到一部分矩阵中元素的计算和查找操作。然而在日常使用中，对应元素的索引也是非常重要的。依然，让我们先从一个脚本开始 ：
```python
import numpy as np
A = np.arange(2, 14).reshape((3, 4))

# array([[ 2, 3, 4, 5]
#        [ 6, 7, 8, 9]
#        [10,11,12,13]])

print(np.argmin(A))    # 0
print(np.argmax(A))    # 11
```

# numpy 的几种基本运算
其中的 argmin() 和 argmax() 两个函数分别对应着求矩阵中最小元素和最大元素的索引。相应的，在矩阵的12个元素中，最小值即2，对应索引0，最大值为13，对应索引为11。
如果需要计算统计中的均值，可以利用下面的方式，将整个矩阵的均值求出来：
```python
print(np.mean(A))        # 7.5
print(np.average(A))     # 7.5
```

仿照着前一节中dot() 的使用法则，mean()函数还有另外一种写法：
```python
print(A.mean())          # 7.5
```

同样的，我们可以写出求解中位数的函数：
```python
print(A.median())       # 7.5
```

另外，和matlab中的cumsum()累加函数类似，Numpy中也具有cumsum()函数，其用法如下：
```python
print(np.cumsum(A))
# [2 5 9 14 20 27 35 44 54 65 77 90]
```

在cumsum()函数中：生成的每一项矩阵元素均是从原矩阵首项累加到对应项的元素之和。比如元素9，在cumsum()生成的矩阵中序号为3，即原矩阵中2，3，4三个元素的和。
相应的有累差运算函数：
```python
print(np.diff(A))
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]]
```
该函数计算的便是每一行中后一项与前一项之差。故一个3行4列矩阵通过函数计算得到的矩阵便是3行3列的矩阵。

下面我们介绍一下nonzero()函数：
```python
print(np.nonzero(A))
# (array([0,0,0,0,1,1,1,1,2,2,2,2]),array([0,1,2,3,0,1,2,3,0,1,2,3]))
```
这个函数将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵。

同样的，我们可以对所有元素进行仿照列表一样的排序操作，但这里的排序函数仍然仅针对每一行进行从小到大排序操作：
```python
import numpy as np
A = np.arange(14, 2, -1).reshape((3, 4))

# array([[14, 13, 12, 11],
#       [10,  9,  8,  7],
#       [ 6,  5,  4,  3]])

print(np.sort(A))

# array([[11,12,13,14]
#        [ 7, 8, 9,10]
#        [ 3, 4, 5, 6]])
```

矩阵的转置有两种表示方法：
```python
print(np.transpose(A))
print(A.T)

# array([[14,10, 6]
#        [13, 9, 5]
#        [12, 8, 4]
#        [11, 7, 3]])
# array([[14,10, 6]
#        [13, 9, 5]
#        [12, 8, 4]
#        [11, 7, 3]])
```

特别的，在Numpy中具有clip()函数，例子如下：
```python
print(A)
# array([[14,13,12,11]
#        [10, 9, 8, 7]
#        [ 6, 5, 4, 3]])

print(np.clip(A, 5, 9))
# array([[ 9, 9, 9, 9]
#        [ 9, 9, 8, 7]
#        [ 6, 5, 5, 5]])
```
这个函数的格式是clip(Array, Array_min, Array_max)，顾名思义，Array指的是将要被执行用的矩阵，而后面的最小值最大值则用于让函数判断矩阵中元素是否有比最小值小的或者比最大值大的元素，并将这些指定的元素转换为最小值或者最大值。
实际上每一个Numpy中大多数函数均具有很多变量可以操作，你可以指定行、列甚至某一范围中的元素。更多具体的使用细节请记得查阅Numpy官方英文教材。

# Numpy 索引
# 一维索引
我们都知道，在元素列表或者数组中，我们可以用如同a[2]一样的表示方法，同样的，在Numpy中也有相对应的表示方法：
```python
import numpy as np
A = np.arange(3, 15)
# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
print(A[3])    # 6
```

让我们将矩阵转换为二维的，此时进行同样的操作：
```python
A = np.arange(3, 15).reshape((3, 4))
"""
array([[ 3,  4,  5,  6]
       [ 7,  8,  9, 10]
       [11, 12, 13, 14]])
"""
print(A[2])
# [11 12 13 14]
```

实际上这时的A[2]对应的就是矩阵A中第三行(从0开始算第一行)的所有元素。

# 二维索引
如果你想要表示具体的单个元素，可以仿照上述的例子：
```python
print(A[1][1])      # 8
```

此时对应的元素即A[1][1]，在A中即横纵坐标都为1，第二行第二列的元素，即8（因为计数从0开始）。同样的还有其他的表示方法：
```python
print(A[1, 1])      # 8
```

在Python的 list 中，我们可以利用: 对一定范围内的元素进行切片操作，在Numpy中我们依然可以给出相应的方法：
```python
print(A[1, 1:3])    # [8 9]
```

这一表示形式即针对第二行中第2到第4列元素进行切片输出（不包含第4列）。 此时我们适当的利用for函数进行打印：
```python
for row in A:
    print(row)
"""    
[ 3,  4,  5, 6]
[ 7,  8,  9, 10]
[11, 12, 13, 14]
"""
```

此时它会逐行进行打印操作。如果想进行逐列打印，就需要稍稍变化一下：
```python
for column in A.T:
    print(column)
"""  
[ 3,  7,  11]
[ 4,  8,  12]
[ 5,  9,  13]
[ 6, 10,  14]
"""
```
上述表示方法即对A进行转置，再将得到的矩阵逐行输出即可得到原矩阵的逐列输出。

最后依然说一些关于迭代输出的问题：
```python
import numpy as np
A = np.arange(3, 15).reshape((3, 4))

print(A.flatten())
# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

for item in A.flat:
    print(item)
# 3
# 4
……
# 14
```
这一脚本中的flatten是一个展开性质的函数，将多维的矩阵进行展开成1行的数列。而flat是一个迭代器，本身是一个object属性。

# Numpy array 合并

# np.vstack()
对于一个array的合并，我们可以想到按行、按列等多种方式进行合并。首先先看一个例子：
```python
import numpy as np
A = np.array([1, 1, 1])
B = np.array([2, 2, 2])

print(np.vstack((A, B)))    # vertical stack
"""
[[1,1,1]
 [2,2,2]]
"""
```

vertical stack本身属于一种上下合并，即对括号中的两个整体进行对应操作。此时我们对组合而成的矩阵进行属性探究：
```python
C = np.vstack((A, B))
print(A.shape, C.shape)
# (3,) (2,3)
```

# np.hstack()
利用shape函数可以让我们很容易地知道A和C的属性，从打印出的结果来看，A仅仅是一个拥有3项元素的数组（数列），而合并后得到的C是一个2行3列的矩阵。
介绍完了上下合并，我们来说说左右合并：
```python
D = np.hstack((A, B))       # horizontal stack
print(D)
# [1,1,1,2,2,2]
print(A.shape, D.shape)
# (3,) (6,)
```
通过打印出的结果可以看出：D本身来源于A，B两个数列的左右合并，而且新生成的D本身也是一个含有6项元素的序列。

# np.newaxis()
说完了array的合并，我们稍稍提及一下前一节中转置操作，如果面对如同前文所述的A序列， 转置操作便很有可能无法对其进行转置（因为A并不是矩阵的属性），此时就需要我们借助其他的函数操作进行转置：
```python
print(A[np.newaxis, :])
# [[1 1 1]]
print(A[np.newaxis, :].shape)
# (1,3)
print(A[:, np.newaxis])
"""
[[1]
[1]
[1]]
"""
print(A[:, np.newaxis].shape)
# (3,1)
```
此时我们便将具有3个元素的array转换为了1行3列以及3行1列的矩阵了。

结合着上面的知识，我们把它综合起来：
```python
import numpy as np
A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 2, 2])[:, np.newaxis]

C = np.vstack((A, B))   # vertical stack
D = np.hstack((A, B))   # horizontal stack

print(D)
"""
[[1 2]
[1 2]
[1 2]]
"""
print(A.shape, D.shape)
# (3,1) (3,2)
```

# np.concatenate()

当你的合并操作需要针对多个矩阵或序列时，借助concatenate函数可能会让你使用起来比前述的函数更加方便：
```python
C = np.concatenate((A, B, B, A), axis=0)

print(C)
"""
array([[1],
       [1],
       [1],
       [2],
       [2],
       [2],
       [2],
       [2],
       [2],
       [1],
       [1],
       [1]])
"""

D = np.concatenate((A, B, B, A), axis=1)

print(D)
"""
array([[1, 2, 2, 1],
       [1, 2, 2, 1],
       [1, 2, 2, 1]])
"""
```
axis参数很好的控制了矩阵的纵向或是横向打印，相比较vstack和hstack函数显得更加方便。

# Numpy array 分割

# 创建数据
首先import 模块
```python
import numpy as np
```

建立3行4列的Array
```python
A = np.arange(12).reshape((3, 4))
print(A)
"""
array([[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]])
"""
```

# 纵向分割
```python
print(np.split(A, 2, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""
```

# 横向分割
```python
print(np.split(A, 3, axis=0))
# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
```

# 错误的分割
范例的Array只有4列，只能等量对分，因此输入以上程序代码后Python就会报错。
```python
print(np.split(A, 3, axis=1))
# ValueError: array split does not result in an equal division
```

为了解决这种情况, 我们会有下面这种方式.

# 不等量的分割
在机器学习时经常会需要将数据做不等量的分割，因此解决办法为np.array_split()
```python
print(np.array_split(A, 3, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), array([[ 2],
        [ 6],
        [10]]), array([[ 3],
        [ 7],
        [11]])]
"""
```
成功将Array不等量分割!

# 其他的分割方式
在Numpy里还有np.vsplit()与横np.hsplit()方式可用。
```python
print(np.vsplit(A, 3))  # 等于 print(np.split(A, 3, axis=0))
# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
print(np.hsplit(A, 2))  # 等于 print(np.split(A, 2, axis=1))
"""
[array([[0, 1],
       [4, 5],
       [8, 9]]), array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""
```

# Numpy copy & deep copy

# = 的赋值方式会带有关联性
首先 import numpy 并建立变量, 给变量赋值。
```python
import numpy as np

a = np.arange(4)
# array([0, 1, 2, 3])

b = a
c = a
d = b
```

改变a的第一个值，b、c、d的第一个值也会同时改变。
```python
a[0] = 11
print(a)
# array([11,  1,  2,  3])
```

确认b、c、d是否与a相同。
```python
b is a  # True
c is a  # True
d is a  # True
```

同样更改d的值，a、b、c也会改变。
```python
d[1:3] = [22, 33]   # array([11, 22, 33,  3])
print(a)            # array([11, 22, 33,  3])
print(b)            # array([11, 22, 33,  3])
print(c)            # array([11, 22, 33,  3])
```

# copy() 的赋值方式没有关联性（深拷贝）
```python
b = a.copy()    # deep copy
print(b)        # array([11, 22, 33,  3])
a[3] = 44
print(a)        # array([11, 22, 33, 44])
print(b)        # array([11, 22, 33,  3])
```
此时a与b已经没有关联。

# Pandas 基本介绍

# Numpy 和 Pandas 有什么不同
如果用 python 的列表和字典来作比较, 那么可以说 Numpy 是列表形式的，没有数值标签，而 Pandas 就是字典形式。Pandas是基于Numpy构建的，让Numpy为中心的应用变得更加简单。

要使用pandas，首先需要了解他主要两个数据结构：Series和DataFrame。

# Series
```python
import pandas as pd
import numpy as np
s = pd.Series([1, 3, 6, np.nan, 44, 1])
print(s)
"""
0     1.0
1     3.0
2     6.0
3     NaN
4    44.0
5     1.0
dtype: float64
"""
```
Series的字符串表现形式为：索引在左边，值在右边。由于我们没有为数据指定索引。于是会自动创建一个0到N - 1（N为长度）的整数型索引。

# DataFrame
```python
dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])

print(df)
"""
                   a         b         c         d
2016-01-01 -0.253065 -2.071051 -0.640515  0.613663
2016-01-02 -1.147178  1.532470  0.989255 -0.499761
2016-01-03  1.221656 -2.390171  1.862914  0.778070
2016-01-04  1.473877 -0.046419  0.610046  0.204672
2016-01-05 -1.584752 -0.700592  1.487264 -1.778293
2016-01-06  0.633675 -1.414157 -0.277066 -0.442545
"""
```
DataFrame是一个表格型的数据结构，它包含有一组有序的列，每列可以是不同的值类型（数值，字符串，布尔值等）。DataFrame既有行索引也有列索引， 它可以被看做由Series组成的大字典。

我们可以根据每一个不同的索引来挑选数据, 比如挑选 b 的元素:

    # DataFrame的一些简单运用
```python
print(df['b'])

"""
2016-01-01   -2.071051
2016-01-02    1.532470
2016-01-03   -2.390171
2016-01-04   -0.046419
2016-01-05   -0.700592
2016-01-06   -1.414157
Freq: D, Name: b, dtype: float64
"""
```

我们在创建一组没有给定行标签和列标签的数据 df1:
```python
df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
print(df1)

"""
   0  1   2   3
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
"""
```

这样, 他就会采取默认的从0开始 index. 还有一种生成 df 的方法, 如下 df2:

```python
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

print(df2)

"""
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
"""
```
这种方法能对每一列的数据进行特殊对待. 如果想要查看数据中的类型, 我们可以用 dtype 这个属性:

```python
print(df2.dtypes)

"""
df2.dtypes
A           float64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
dtype: object
"""
```

如果想看对列的序号:
```python
print(df2.index)
# Int64Index([0, 1, 2, 3], dtype='int64')
```

同样, 每种数据的名称也能看到:
```python
print(df2.columns)
# Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')
```

如果只想看所有df2的值:
```python
print(df2.values)

"""
array([[1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo']], dtype=object)
"""
```

想知道数据的总结, 可以用 describe():
```python
df2.describe()
"""
         A    C    D
count  4.0  4.0  4.0
mean   1.0  1.0  3.0
std    0.0  0.0  0.0
min    1.0  1.0  3.0
25%    1.0  1.0  3.0
50%    1.0  1.0  3.0
75%    1.0  1.0  3.0
max    1.0  1.0  3.0
"""
```

如果想翻转数据, transpose:
```python
print(df2.T)

"""                   
0                    1                    2  \
A                    1                    1                    1   
B  2013-01-02 00:00:00  2013-01-02 00:00:00  2013-01-02 00:00:00   
C                    1                    1                    1   
D                    3                    3                    3   
E                 test                train                 test   
F                  foo                  foo                  foo   

                     3  
A                    1  
B  2013-01-02 00:00:00  
C                    1  
D                    3  
E                train  
F                  foo  

"""
```

如果想对数据的 index 进行排序并输出:
```python
print(df2.sort_index(axis=1, ascending=False))

"""
     F      E  D    C          B    A
0  foo   test  3  1.0 2013-01-02  1.0
1  foo  train  3  1.0 2013-01-02  1.0
2  foo   test  3  1.0 2013-01-02  1.0
3  foo  train  3  1.0 2013-01-02  1.0
"""
```

如果是对数据 值 排序输出:
```python
print(df2.sort_values(by='B'))

"""
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
"""
```

# Pandas选择数据
我们建立了一个 6X4 的矩阵数据。
```python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

"""
             A   B   C   D
2013-01-01   0   1   2   3
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""
```

# 简单的筛选
如果我们想选取DataFrame中的数据，下面描述了两种途径, 他们都能达到同一个目的：
```python
print(df['A'])
print(df.A)

"""
2013-01-01     0
2013-01-02     4
2013-01-03     8
2013-01-04    12
2013-01-05    16
2013-01-06    20
Freq: D, Name: A, dtype: int64
"""
```

让选择跨越多行或多列：
```python
print(df[0:3])

"""
            A  B   C   D
2013-01-01  0  1   2   3
2013-01-02  4  5   6   7
2013-01-03  8  9  10  11
"""

print(df['20130102':'20130104'])

"""
A   B   C   D
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
"""
```
如果df[3:3]将会是一个空对象。后者选择20130102到20130104标签之间的数据，并且包括这两个标签。

# 根据标签 loc
同样我们可以使用标签来选择数据 loc, 本例子主要通过标签名字选择某一行数据， 或者通过选择某行或者所有行（: 代表所有行）然后选其中某一列或几列数据:
```python
print(df.loc['20130102'])
"""
A    4
B    5
C    6
D    7
Name: 2013-01-02 00:00:00, dtype: int64
"""

print(df.loc[:, ['A', 'B']])
"""
             A   B
2013-01-01   0   1
2013-01-02   4   5
2013-01-03   8   9
2013-01-04  12  13
2013-01-05  16  17
2013-01-06  20  21
"""

print(df.loc['20130102', ['A', 'B']])
"""
A    4
B    5
Name: 2013-01-02 00:00:00, dtype: int64
"""
```

# 根据序列 iloc

```python
print(df.iloc[3, 1])
# 13

print(df.iloc[3:5, 1:3])
"""
             B   C
2013-01-04  13  14
2013-01-05  17  18
"""

print(df.iloc[[1, 3, 5], 1:3])
"""
             B   C
2013-01-02   5   6
2013-01-04  13  14
2013-01-06  21  22

"""
```

在这里我们可以通过位置选择在不同情况下所需要的数据, 例如选某一个，连续选或者跨行选等操作。

# 根据混合的这两种 ix
当然我们可以采用混合选择 ix, 其中选择’A’和’C’的两列，并选择前三行的数据。
```python
print(df.ix[:3, ['A', 'C']])
"""
            A   C
2013-01-01  0   2
2013-01-02  4   6
2013-01-03  8  10
"""
```

# 通过判断的筛选
最后我们可以采用判断指令(Boolean indexing) 进行选择. 我们可以约束某项条件然后选择出当前所有数据.
```python
print(df[df.A > 8])
"""
             A   B   C   D
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""
```
下节我们将会讲到Pandas中如何设置值。

# Pandas 设置值
# 创建数据
我们可以根据自己的需求, 用 pandas 进行更改数据里面的值, 或者加上一些空的, 或者有数值的列。
首先建立了一个 6X4 的矩阵数据。

```python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

"""
             A   B   C   D
2013-01-01   0   1   2   3
2013-01-02   4   5   6   7
2013-01-03   8   9  10  11
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""
```

# 根据位置设置 loc 和 iloc
我们可以利用索引或者标签确定需要修改值的位置。
```python
df.iloc[2, 2] = 1111
df.loc['20130101', 'B'] = 2222

"""
             A     B     C   D
2013-01-01   0  2222     2   3
2013-01-02   4     5     6   7
2013-01-03   8     9  1111  11
2013-01-04  12    13    14  15
2013-01-05  16    17    18  19
2013-01-06  20    21    22  23
"""
```

## 根据条件设置
如果现在的判断条件是这样, 我们想要更改B中的数, 而更改的位置是取决于 A 的. 对于A大于4的位置. 更改B在相应位置上的数为0.
```Python
df.B[df.A>4] = 0
"""
                A     B     C   D
2013-01-01   0  2222     2   3
2013-01-02   4     5     6   7
2013-01-03   8     0  1111  11
2013-01-04  12     0    14  15
2013-01-05  16     0    18  19
2013-01-06  20     0    22  23 
"""
```

## 按行或列设置
如果对整列做批处理, 加上一列 ‘F’, 并将 F 列全改为 NaN, 如下:
```Python
df['F'] = np.nan
"""
             A     B     C   D   F
2013-01-01   0  2222     2   3 NaN
2013-01-02   4     5     6   7 NaN
2013-01-03   8     0  1111  11 NaN
2013-01-04  12     0    14  15 NaN
2013-01-05  16     0    18  19 NaN
2013-01-06  20     0    22  23 NaN
"""
```

## 添加数据
用上面的方法也可以加上 Series 序列（但是长度必须对齐）。
```Python
df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101',periods=6)) 
"""
             A     B     C   D   F  E
2013-01-01   0  2222     2   3 NaN  1
2013-01-02   4     5     6   7 NaN  2
2013-01-03   8     0  1111  11 NaN  3
2013-01-04  12     0    14  15 NaN  4
2013-01-05  16     0    18  19 NaN  5
2013-01-06  20     0    22  23 NaN  6
"""
```
这样我们大概学会了如何对DataFrame中在自己想要的地方赋值或者增加数据。下次课会将pandas如何处理丢失数据的过程。
