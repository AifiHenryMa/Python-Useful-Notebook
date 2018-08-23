# 一、Numpy & Pandas简介
## 1.1、Why Numpy & Pandas?
### 1.1.1、应用
- 数据分析
- 机器学习
- 深度学习

### 1.1.2、为什么使用numpy & pandas
- 运算速度快：numpy 和 pandas 都是采用 C 语言编写, pandas 又是基于 numpy, 是 numpy 的升级版本。
- 消耗资源少：采用的是矩阵运算，会比 python 自带的字典或者列表快好多。

## 1.2、Numpy 和 Pandas 安装
### 1.2.3、numpy 安装
- sudo apt - get install python - numpy
- 使用 python 3 + : pip3 install numpy
- 使用 python 2 + : pip install numpy

### 1.2.4、pandas 安装
- sudo apt - get install python - pandas
- 使用 python 3 + : pip3 install pandas
- 使用 python 2 + : pip install pandas

# 二、Numpy学习
## 2.1、Numpy 属性
这次我们会介绍几种 numpy 的属性:
- ndim: 维度
- shape: 行数和列数
- size: 元素个数

使用numpy首先要导入模块
```Python
import numpy as np  # 为了方便使用numpy 采用np简写
```

列表转化为矩阵：
```Python
array = np.array([[1, 2, 3], [2, 3, 4]])  # 列表转化为矩阵
print(array)
"""
array([[1, 2, 3],
       [2, 3, 4]])
"""
```

### 2.1.1、numpy 的几种属性
接下来我们看看这几种属性的结果：

```Python
print('number of dim:', array.ndim)  # 维度
# number of dim: 2

print('shape :', array.shape)    # 行数和列数
# shape : (2, 3)

print('size:', array.size)   # 元素个数
# size: 6
```

## 2.2、Numpy 的创建 array
### 2.2.1、关键字
- array：创建数组
- dtype：指定数据类型
- zeros：创建数据全为0
- ones：创建数据全为1
- empty：创建数据接近0
- arrange：按指定范围创建数据
- linspace：创建线段

### 2.2.2、创建数组

```Python
a = np.array([2, 23, 4])  # list 1d
print(a)
# [2 23 4]
```

### 2.2.3、指定数据 dtype

```Python
a = np.array([2, 23, 4], dtype=np.int)
print(a.dtype)
# int 64
```

```Python
a = np.array([2, 23, 4], dtype=np.int32)
print(a.dtype)
# int32
```

```Python
a = np.array([2, 23, 4], dtype=np.float)
print(a.dtype)
# float64
```

```Python
a = np.array([2, 23, 4], dtype=np.float32)
print(a.dtype)
# float32
```

### 2.2.4、创建特定数据

```Python
a = np.array([[2, 23, 4], [2, 32, 4]])  # 2d 矩阵 2行3列
print(a)
"""
[[ 2 23  4]
 [ 2 32  4]]
"""
```

创建全零数组
```Python
a = np.zeros((3, 4))  # 数据全为0，3行4列
"""
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
"""
```

创建全一数组, 同时也能指定这些特定数据的 dtype:
```Python
a = np.ones((3, 4), dtype=np.int)   # 数据为1，3行4列
"""
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
"""
```

创建全空数组, 其实每个值都是接近于零的数:
```Python
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
```Python
a = np.arange(10, 20, 2)  # 10-19 的数据，2步长
"""
array([10, 12, 14, 16, 18])
"""
```

使用 reshape 改变数据的形状
```Python
a = np.arange(12).reshape((3, 4))    # 3行4列，0到11
"""
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
"""
```

用 linspace 创建线段型数据:
```Python
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
```Python
a = np.linspace(1, 10, 20).reshape((5, 4))  # 更改shape
"""
array([[  1.        ,   1.47368421,   1.94736842,   2.42105263],
       [  2.89473684,   3.36842105,   3.84210526,   4.31578947],
       [  4.78947368,   5.26315789,   5.73684211,   6.21052632],
       [  6.68421053,   7.15789474,   7.63157895,   8.10526316],
       [  8.57894737,   9.05263158,   9.52631579,  10.        ]])
"""
```

## 2.3、Numpy 基础运算1

让我们从一个脚本开始了解相应的计算以及表示形式 ：

```Python
import numpy as np
a = np.array([10, 20, 30, 40])   # array([10, 20, 30, 40])
b = np.arange(4)              # array([0, 1, 2, 3])
```

### 2.3.1、numpy 的几种基本运算

减法，加法，乘法：
```Python
c = a - b  # array([10, 19, 28, 37])
c = a + b   # array([10, 21, 32, 43])
c = a * b   # array([  0,  20,  60, 120])
```

有所不同的是，在Numpy中，想要求出矩阵中各个元素的乘方需要依赖双星符号 **，以二次方举例，即：
```Python
c = b**2  # array([0, 1, 4, 9])
```

另外，Numpy中具有很多的数学函数工具，比如三角函数等，当我们需要对矩阵中每一项元素进行函数运算时，可以很简便的调用它们（以sin函数为例）：
```Python
c = 10 * np.sin(a)
# array([-5.44021111,  9.12945251, -9.88031624,  7.4511316 ])
```

除了函数应用外，在脚本中对print函数进行一些修改可以进行逻辑判断：
```Python
print(b < 3)
# array([ True,  True,  True, False], dtype=bool)
```

此时由于进行逻辑判断，返回的是一个bool类型的矩阵，即对满足要求的返回True，不满足的返回False。上述程序执行后得到的结果是[True True True False]。 需要注意的是，如果想要执行是否相等的判断， 依然需要输入 == 而不是 = 来完成相应的逻辑判断。

上述运算均是建立在一维矩阵，即只有一行的矩阵上面的计算，如果我们想要对多行多维度的矩阵进行操作，需要对开始的脚本进行一些修改：
```Python
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
```Python
c_dot = np.dot(a, b)
# array([[2, 4],
#       [2, 3]])
```

除此之外还有另外的一种关于dot的表示方法，即：
```Python
c_dot_2 = a.dot(b)
# array([[2, 4],
#       [2, 3]])
```

下面我们将重新定义一个脚本, 来看看关于 sum(), min(), max()的使用：
```Python
import numpy as np
a = np.random.random((2, 4))
print(a)
# array([[ 0.94692159,  0.20821798,  0.35339414,  0.2805278 ],
#       [ 0.04836775,  0.04023552,  0.44091941,  0.21665268]])
```

因为是随机生成数字, 所以你的结果可能会不一样. 在第二行中对a的操作是令a中生成一个2行4列的矩阵，且每一元素均是来自从0到1的随机数。 在这个随机生成的矩阵中，我们可以对元素进行求和以及寻找极值的操作，具体如下：
```Python
np.sum(a)   # 4.4043622002745959
np.min(a)   # 0.23651223533671784
np.max(a)   # 0.90438450240606416
```

对应的便是对矩阵中所有元素进行求和，寻找最小值，寻找最大值的操作。 可以通过print()函数对相应值进行打印检验。
如果你需要对行或者列进行查找运算，就需要在上述代码中为 axis 进行赋值。 当axis的值为0的时候，将会以列作为查找单元， 当axis的值为1的时候，将会以行作为查找单元。
为了更加清晰，在刚才的例子中我们继续进行查找：
```Python
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

## 2.4、Numpy 基础运算2

通过上一节的学习，我们可以了解到一部分矩阵中元素的计算和查找操作。然而在日常使用中，对应元素的索引也是非常重要的。依然，让我们先从一个脚本开始 ：
```Python
import numpy as np
A = np.arange(2, 14).reshape((3, 4))

# array([[ 2, 3, 4, 5]
#        [ 6, 7, 8, 9]
#        [10,11,12,13]])

print(np.argmin(A))    # 0
print(np.argmax(A))    # 11
```

### 2.4.1、numpy 的几种基本运算
其中的 argmin() 和 argmax() 两个函数分别对应着求矩阵中最小元素和最大元素的索引。相应的，在矩阵的12个元素中，最小值即2，对应索引0，最大值为13，对应索引为11。
如果需要计算统计中的均值，可以利用下面的方式，将整个矩阵的均值求出来：
```Python
print(np.mean(A))        # 7.5
print(np.average(A))     # 7.5
```

仿照着前一节中dot() 的使用法则，mean()函数还有另外一种写法：
```Python
print(A.mean())          # 7.5
```

同样的，我们可以写出求解中位数的函数：
```Python
print(A.median())       # 7.5
```

另外，和matlab中的cumsum()累加函数类似，Numpy中也具有cumsum()函数，其用法如下：
```Python
print(np.cumsum(A))
# [2 5 9 14 20 27 35 44 54 65 77 90]
```

在cumsum()函数中：生成的每一项矩阵元素均是从原矩阵首项累加到对应项的元素之和。比如元素9，在cumsum()生成的矩阵中序号为3，即原矩阵中2，3，4三个元素的和。
相应的有累差运算函数：
```Python
print(np.diff(A))
# [[1 1 1]
#  [1 1 1]
#  [1 1 1]]
```
该函数计算的便是每一行中后一项与前一项之差。故一个3行4列矩阵通过函数计算得到的矩阵便是3行3列的矩阵。

下面我们介绍一下nonzero()函数：
```Python
print(np.nonzero(A))
# (array([0,0,0,0,1,1,1,1,2,2,2,2]),array([0,1,2,3,0,1,2,3,0,1,2,3]))
```
这个函数将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵。

同样的，我们可以对所有元素进行仿照列表一样的排序操作，但这里的排序函数仍然仅针对每一行进行从小到大排序操作：
```Python
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
```Python
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
```Python
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

## 2.5、Numpy 索引
### 2.5.1、一维索引
我们都知道，在元素列表或者数组中，我们可以用如同a[2]一样的表示方法，同样的，在Numpy中也有相对应的表示方法：
```Python
import numpy as np
A = np.arange(3, 15)
# array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
print(A[3])    # 6
```

让我们将矩阵转换为二维的，此时进行同样的操作：
```Python
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

### 2.5.2、二维索引
如果你想要表示具体的单个元素，可以仿照上述的例子：
```Python
print(A[1][1])      # 8
```

此时对应的元素即A[1][1]，在A中即横纵坐标都为1，第二行第二列的元素，即8（因为计数从0开始）。同样的还有其他的表示方法：
```Python
print(A[1, 1])      # 8
```

在Python的 list 中，我们可以利用: 对一定范围内的元素进行切片操作，在Numpy中我们依然可以给出相应的方法：
```Python
print(A[1, 1:3])    # [8 9]
```

这一表示形式即针对第二行中第2到第4列元素进行切片输出（不包含第4列）。 此时我们适当的利用for函数进行打印：
```Python
for row in A:
    print(row)
"""    
[ 3,  4,  5, 6]
[ 7,  8,  9, 10]
[11, 12, 13, 14]
"""
```

此时它会逐行进行打印操作。如果想进行逐列打印，就需要稍稍变化一下：
```Python
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
```Python
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

## 2.6、Numpy array 合并

### 2.6.1、np.vstack()
对于一个array的合并，我们可以想到按行、按列等多种方式进行合并。首先先看一个例子：
```Python
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
```Python
C = np.vstack((A, B))
print(A.shape, C.shape)
# (3,) (2,3)
```

### 2.6.2、np.hstack()
利用shape函数可以让我们很容易地知道A和C的属性，从打印出的结果来看，A仅仅是一个拥有3项元素的数组（数列），而合并后得到的C是一个2行3列的矩阵。
介绍完了上下合并，我们来说说左右合并：
```Python
D = np.hstack((A, B))       # horizontal stack
print(D)
# [1,1,1,2,2,2]
print(A.shape, D.shape)
# (3,) (6,)
```
通过打印出的结果可以看出：D本身来源于A，B两个数列的左右合并，而且新生成的D本身也是一个含有6项元素的序列。

### 2.6.3、np.newaxis()
说完了array的合并，我们稍稍提及一下前一节中转置操作，如果面对如同前文所述的A序列， 转置操作便很有可能无法对其进行转置（因为A并不是矩阵的属性），此时就需要我们借助其他的函数操作进行转置：
```Python
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
```Python
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

### 2.6.4、np.concatenate()

当你的合并操作需要针对多个矩阵或序列时，借助concatenate函数可能会让你使用起来比前述的函数更加方便：
```Python
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

## 2.7、Numpy array 分割

### 2.7.1、创建数据
首先import 模块
```Python
import numpy as np
```

建立3行4列的Array
```Python
A = np.arange(12).reshape((3, 4))
print(A)
"""
array([[ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]])
"""
```

### 2.7.2、纵向分割
```Python
print(np.split(A, 2, axis=1))
"""
[array([[0, 1],
        [4, 5],
        [8, 9]]), array([[ 2,  3],
        [ 6,  7],
        [10, 11]])]
"""
```

### 2.7.3、横向分割
```Python
print(np.split(A, 3, axis=0))
# [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
```

### 2.7.4、错误的分割
范例的Array只有4列，只能等量对分，因此输入以上程序代码后Python就会报错。
```Python
print(np.split(A, 3, axis=1))
# ValueError: array split does not result in an equal division
```

为了解决这种情况, 我们会有下面这种方式.

### 2.7.5、不等量的分割
在机器学习时经常会需要将数据做不等量的分割，因此解决办法为np.array_split()
```Python
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

### 2.7.6、其他的分割方式
在Numpy里还有np.vsplit()与横np.hsplit()方式可用。
```Python
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

## 2.8、Numpy copy & deep copy

### 2.8.1、= 的赋值方式会带有关联性
首先 import numpy 并建立变量, 给变量赋值。
```Python
import numpy as np

a = np.arange(4)
# array([0, 1, 2, 3])

b = a
c = a
d = b
```

改变a的第一个值，b、c、d的第一个值也会同时改变。
```Python
a[0] = 11
print(a)
# array([11,  1,  2,  3])
```

确认b、c、d是否与a相同。
```Python
b is a  # True
c is a  # True
d is a  # True
```

同样更改d的值，a、b、c也会改变。
```Python
d[1:3] = [22, 33]   # array([11, 22, 33,  3])
print(a)            # array([11, 22, 33,  3])
print(b)            # array([11, 22, 33,  3])
print(c)            # array([11, 22, 33,  3])
```

### copy() 的赋值方式没有关联性（深拷贝）
```Python
b = a.copy()    # deep copy
print(b)        # array([11, 22, 33,  3])
a[3] = 44
print(a)        # array([11, 22, 33, 44])
print(b)        # array([11, 22, 33,  3])
```
此时a与b已经没有关联。

# 三、Pandas学习
## 3.1、Pandas 基本介绍
### 3.1.1、Numpy 和 Pandas 有什么不同
如果用 python 的列表和字典来作比较, 那么可以说 Numpy 是列表形式的，没有数值标签，而 Pandas 就是字典形式。Pandas是基于Numpy构建的，让Numpy为中心的应用变得更加简单。

要使用pandas，首先需要了解他主要两个数据结构：Series和DataFrame。

### 3.1.2、Series
```Python
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

### 3.1.3、DataFrame
```Python
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

### 3.1.4、DataFrame的一些简单运用
```Python
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
```Python
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

```Python
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

```Python
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
```Python
print(df2.index)
# Int64Index([0, 1, 2, 3], dtype='int64')
```

同样, 每种数据的名称也能看到:
```Python
print(df2.columns)
# Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')
```

如果只想看所有df2的值:
```Python
print(df2.values)

"""
array([[1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo']], dtype=object)
"""
```

想知道数据的总结, 可以用 describe():
```Python
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
```Python
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
```Python
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
```Python
print(df2.sort_values(by='B'))

"""
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
"""
```

## 3.2、Pandas选择数据
我们建立了一个 6X4 的矩阵数据。
```Python
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

### 3.2.1、简单的筛选
如果我们想选取DataFrame中的数据，下面描述了两种途径, 他们都能达到同一个目的：
```Python
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
```Python
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

### 3.2.2、根据标签 loc
同样我们可以使用标签来选择数据 loc, 本例子主要通过标签名字选择某一行数据， 或者通过选择某行或者所有行（: 代表所有行）然后选其中某一列或几列数据:
```Python
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

### 3.2.3、根据序列 iloc
```Python
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

### 3.2.4、根据混合的这两种 ix
当然我们可以采用混合选择 ix, 其中选择’A’和’C’的两列，并选择前三行的数据。
```Python
print(df.ix[:3, ['A', 'C']])
"""
            A   C
2013-01-01  0   2
2013-01-02  4   6
2013-01-03  8  10
"""
```

### 3.2.5、通过判断的筛选
最后我们可以采用判断指令(Boolean indexing) 进行选择. 我们可以约束某项条件然后选择出当前所有数据.
```Python
print(df[df.A > 8])
"""
             A   B   C   D
2013-01-04  12  13  14  15
2013-01-05  16  17  18  19
2013-01-06  20  21  22  23
"""
```
下节我们将会讲到Pandas中如何设置值。

## 3.3、Pandas 设置值
### 3.3.1、创建数据
我们可以根据自己的需求, 用 pandas 进行更改数据里面的值, 或者加上一些空的, 或者有数值的列。
首先建立了一个 6X4 的矩阵数据。

```Python
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

### 3.3.2、根据位置设置 loc 和 iloc
我们可以利用索引或者标签确定需要修改值的位置。
```Python
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

### 3.3.3、根据条件设置
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

### 3.3.4、按行或列设置
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

### 3.4.5、添加数据
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

## 3.4、Pandas 处理丢失数据

### 3.4.1、创建含 NaN 的矩阵
有时候我们导入或处理数据, 会产生一些空的或者是 NaN 数据,如何删除或者是填补这些 NaN 数据就是我们今天所要提到的内容.
建立了一个6X4的矩阵数据并且把两个位置置为空.
```Python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
"""
             A     B     C   D
2013-01-01   0   NaN   2.0   3
2013-01-02   4   5.0   NaN   7
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""
```

### 3.4.2、pd.dropna()
如果想直接去掉有 NaN 的行或列, 可以使用 dropna
```Python
df.dropna(
    axis=0,     # 0: 对行进行操作; 1: 对列进行操作
    how='any'   # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop 
    ) 
"""
             A     B     C   D
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""
```

### 3.4.3、pd.fillna()
如果是将 NaN 的值用其他值代替, 比如代替成 0:
```Python
df.fillna(value=0)
"""
             A     B     C   D
2013-01-01   0   0.0   2.0   3
2013-01-02   4   5.0   0.0   7
2013-01-03   8   9.0  10.0  11
2013-01-04  12  13.0  14.0  15
2013-01-05  16  17.0  18.0  19
2013-01-06  20  21.0  22.0  23
"""
```

### 3.4.4、pd.isnull()

判断是否有缺失数据 NaN, 为 True 表示缺失数据:
```Python
df.isnull() 
"""
                A      B      C      D
2013-01-01  False   True  False  False
2013-01-02  False  False   True  False
2013-01-03  False  False  False  False
2013-01-04  False  False  False  False
2013-01-05  False  False  False  False
2013-01-06  False  False  False  False
"""
```

检测在数据中是否存在 NaN, 如果存在就返回 True:
```Python
np.any(df.isnull()) == True  
# True
```
下次课会将pandas如何导入导出数据的过程。

## 3.5、Pandas导入导出
### 3.5.1、要点
pandas可以读取与存取的资料格式有很多种，像csv、excel、json、html与pickle等…， 详细请看官方说明文件

### 3.5.2、读取csv
```
student.csv内容:
Student ID,name ,age,gender
1100,Kelly,22,Female
1101,Clo,21,Female
1102,Tilly,22,Female
1103,Tony,24,Male
1104,David,20,Male
1105,Catty,22,Female
1106,M,3,Female
1107,N,43,Male
1108,A,13,Male
1109,S,12,Male
1110,David,33,Male
1111,Dw,3,Female
1112,Q,23,Male
1113,W,21,Female
```

```Python
import pandas as pd #加载模块
#读取csv
data = pd.read.csv("student.csv")

#打印出data
print(data)
```

### 3.5.3、将资料存取成pickle 
```Python
data.to_pickle('student.pickle')
```

## 3.6、Pandas 合并 concat

### 3.6.1、要点
pandas处理多组数据的时候往往会要用到数据的合并处理,使用concat是一种基本的合并方式.而且concat中有很多参数可以调整,合并成你想要的数据形式.

### 3.6.2、axis(合并方向)
axis=0是预设值，因此未设定任何参数时，函数默认axis=0。
```Python
import pandas as pd
import numpy as np

#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

#concat纵向合并
res = pd.concat([df1, df2, df3], axis=0)

#打印结果
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 0  1.0  1.0  1.0  1.0
# 1  1.0  1.0  1.0  1.0
# 2  1.0  1.0  1.0  1.0
# 0  2.0  2.0  2.0  2.0
# 1  2.0  2.0  2.0  2.0
# 2  2.0  2.0  2.0  2.0
```
仔细观察会发现结果的index是0, 1, 2, 0, 1, 2, 0, 1, 2，若要将index重置，请看例子二。

### 3.6.3、ignore_index (重置 index) 
```Python
#承上一个例子，并将index_ignore设定为True
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

#打印结果
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0
# 6  2.0  2.0  2.0  2.0
# 7  2.0  2.0  2.0  2.0
# 8  2.0  2.0  2.0  2.0
```
结果的index变0, 1, 2, 3, 4, 5, 6, 7, 8。

### 3.6.4、join (合并方式) 
join='outer'为预设值，因此未设定任何参数时，函数默认join='outer'。此方式是依照column来做纵向合并，有相同的column上下合并在一起，其他独自的column个自成列，原本没有值的位置皆以NaN填充。
```Python
import pandas as pd
import numpy as np

#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

#纵向"外"合并df1与df2
res = pd.concat([df1, df2], axis=0, join='outer')

print(res)
#     a    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN
# 2  0.0  0.0  0.0  0.0  NaN
# 3  0.0  0.0  0.0  0.0  NaN
# 2  NaN  1.0  1.0  1.0  1.0
# 3  NaN  1.0  1.0  1.0  1.0
# 4  NaN  1.0  1.0  1.0  1.0
```

原理同上个例子的说明，但只有相同的column合并在一起，其他的会被抛弃。

```Python
#承上一个例子

#纵向"内"合并df1与df2
res = pd.concat([df1, df2], axis=0, join='inner')

#打印结果
print(res)
#     b    c    d
# 1  0.0  0.0  0.0
# 2  0.0  0.0  0.0
# 3  0.0  0.0  0.0
# 2  1.0  1.0  1.0
# 3  1.0  1.0  1.0
# 4  1.0  1.0  1.0

#重置index并打印结果
res = pd.concat([df1, df2], axis=0, join='inner', ignore_index=True)
print(res)
#     b    c    d
# 0  0.0  0.0  0.0
# 1  0.0  0.0  0.0
# 2  0.0  0.0  0.0
# 3  1.0  1.0  1.0
# 4  1.0  1.0  1.0
# 5  1.0  1.0  1.0
```

### 3.6.5、join_axes (依照 axes 合并)
```Python
import pandas as pd
import numpy as np

#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])

#依照`df1.index`进行横向合并
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])

#打印结果
print(res)
#     a    b    c    d    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
# 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0

#移除join_axes，并打印结果
res = pd.concat([df1, df2], axis=1)
print(res)
#     a    b    c    d    b    c    d    e
# 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
# 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
# 4  NaN  NaN  NaN  NaN  1.0  1.0  1.0  1.0
```

### 3.6.6、append (添加数据)
append只有纵向合并，没有横向合并。
```Python
import pandas as pd
import numpy as np

#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])

#将df2合并到df1的下面，以及重置index，并打印出结果
res = df1.append(df2, ignore_index=True)
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0

#合并多个df，将df2与df3合并至df1的下面，以及重置index，并打印出结果
res = df1.append([df2, df3], ignore_index=True)
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  1.0  1.0  1.0
# 4  1.0  1.0  1.0  1.0
# 5  1.0  1.0  1.0  1.0
# 6  1.0  1.0  1.0  1.0
# 7  1.0  1.0  1.0  1.0
# 8  1.0  1.0  1.0  1.0

#合并series，将s1合并至df1，以及重置index，并打印出结果
res = df1.append(s1, ignore_index=True)
print(res)
#     a    b    c    d
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 3  1.0  2.0  3.0  4.0
```

## 3.7、Pandas合并merge
### 3.7.1、要点
pandas中的merge和concat类似,但主要是用于两组有key column的数据,统一索引的数据. 通常也被用在Database的处理当中.

### 3.7.2、依据一组key合并
```Python
import pandas as pd

#定义资料集并打印出
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
#    A   B key
# 0  A0  B0  K0
# 1  A1  B1  K1
# 2  A2  B2  K2
# 3  A3  B3  K3

print(right)
#    C   D key
# 0  C0  D0  K0
# 1  C1  D1  K1
# 2  C2  D2  K2
# 3  C3  D3  K3

#依据key column合并，并打印出
res = pd.merge(left, right, on='key')

print(res)
     A   B key   C   D
# 0  A0  B0  K0  C0  D0
# 1  A1  B1  K1  C1  D1
# 2  A2  B2  K2  C2  D2
# 3  A3  B3  K3  C3  D3
```

## 3.8、Pandas合并merge
### 3.8.1、要点
pandas中的merge和concat类似,但主要是用于两组有key column的数据,统一索引的数据. 通常也被用在Database的处理当中.

### 3.8.2、依据一组key合并
```Python
import pandas as pd

#定义资料集并打印出
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
#    A   B key
# 0  A0  B0  K0
# 1  A1  B1  K1
# 2  A2  B2  K2
# 3  A3  B3  K3

print(right)
#    C   D key
# 0  C0  D0  K0
# 1  C1  D1  K1
# 2  C2  D2  K2
# 3  C3  D3  K3

#依据key column合并，并打印出
res = pd.merge(left, right, on='key')

print(res)
     A   B key   C   D
# 0  A0  B0  K0  C0  D0
# 1  A1  B1  K1  C1  D1
# 2  A2  B2  K2  C2  D2
# 3  A3  B3  K3  C3  D3
```

### 3.8.3、依据两组key合并
合并时有4种方法how = ['left', 'right', 'outer', 'inner']，预设值how='inner'。
```Python
import pandas as pd

#定义资料集并打印出
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
#    A   B key1 key2
# 0  A0  B0   K0   K0
# 1  A1  B1   K0   K1
# 2  A2  B2   K1   K0
# 3  A3  B3   K2   K1

print(right)
#    C   D key1 key2
# 0  C0  D0   K0   K0
# 1  C1  D1   K1   K0
# 2  C2  D2   K1   K0
# 3  C3  D3   K2   K0

#依据key1与key2 columns进行合并，并打印出四种结果['left', 'right', 'outer', 'inner']
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
print(res)
#    A   B key1 key2   C   D
# 0  A0  B0   K0   K0  C0  D0
# 1  A2  B2   K1   K0  C1  D1
# 2  A2  B2   K1   K0  C2  D2

res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
print(res)
#     A    B key1 key2    C    D
# 0   A0   B0   K0   K0   C0   D0
# 1   A1   B1   K0   K1  NaN  NaN
# 2   A2   B2   K1   K0   C1   D1
# 3   A2   B2   K1   K0   C2   D2
# 4   A3   B3   K2   K1  NaN  NaN
# 5  NaN  NaN   K2   K0   C3   D3

res = pd.merge(left, right, on=['key1', 'key2'], how='left')
print(res)
#    A   B key1 key2    C    D
# 0  A0  B0   K0   K0   C0   D0
# 1  A1  B1   K0   K1  NaN  NaN
# 2  A2  B2   K1   K0   C1   D1
# 3  A2  B2   K1   K0   C2   D2
# 4  A3  B3   K2   K1  NaN  NaN

res = pd.merge(left, right, on=['key1', 'key2'], how='right')
print(res)
#     A    B key1 key2   C   D
# 0   A0   B0   K0   K0  C0  D0
# 1   A2   B2   K1   K0  C1  D1
# 2   A2   B2   K1   K0  C2  D2
# 3  NaN  NaN   K2   K0  C3  D3
```

### 3.8.4、Indicator
indicator=True会将合并的记录放在新的一列。
```Python
import pandas as pd

#定义资料集并打印出
df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})

print(df1)
#   col1 col_left
# 0     0        a
# 1     1        b

print(df2)
#   col1  col_right
# 0     1          2
# 1     2          2
# 2     2          2

# 依据col1进行合并，并启用indicator=True，最后打印出
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
print(res)
#   col1 col_left  col_right      _merge
# 0   0.0        a        NaN   left_only
# 1   1.0        b        2.0        both
# 2   2.0      NaN        2.0  right_only
# 3   2.0      NaN        2.0  right_only

# 自定indicator column的名称，并打印出
res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
print(res)
#   col1 col_left  col_right indicator_column
# 0   0.0        a        NaN        left_only
# 1   1.0        b        2.0             both
# 2   2.0      NaN        2.0       right_only
# 3   2.0      NaN        2.0       right_only
```

### 3.8.5、依据index合并
```Python
import pandas as pd

#定义资料集并打印出
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])

print(left)
#     A   B
# K0  A0  B0
# K1  A1  B1
# K2  A2  B2

print(right)
#     C   D
# K0  C0  D0
# K2  C2  D2
# K3  C3  D3

#依据左右资料集的index进行合并，how='outer',并打印出
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
print(res)
#      A    B    C    D
# K0   A0   B0   C0   D0
# K1   A1   B1  NaN  NaN
# K2   A2   B2   C2   D2
# K3  NaN  NaN   C3   D3

#依据左右资料集的index进行合并，how='inner',并打印出
res = pd.merge(left, right, left_index=True, right_index=True, how='inner')
print(res)
#     A   B   C   D
# K0  A0  B0  C0  D0
# K2  A2  B2  C2  D2
```

### 3.8.6、解决overlapping的问题
```Python
import pandas as pd

#定义资料集
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})

#使用suffixes解决overlapping的问题
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(res)
#    age_boy   k  age_girl
# 0        1  K0         4
# 1        1  K0         5
```

## 3.9 Pandas plot出图
这次我们讲如何将数据可视化. 首先import我们需要用到的模块，除了 pandas，我们也需要使用 numpy 生成一些数据，这节里使用的 matplotlib 仅仅是用来 show 图片的, 即 plt.show()。
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
今天我们主要是学习如何plot data

### 3.9.1 创建一个Series
这是一个线性的数据，我们随机生成1000个数据，Series 默认的 index 就是从0开始的整数，但是这里我显式赋值以便让大家看的更清楚
```Python
import matplotlib.pyplot as plt
# 随机生成1000个数据
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
 
# 为了方便观看效果, 我们累加这个数据
data.cumsum()

# pandas 数据可以直接观看其可视化形式
data.plot()

plt.show()
```

就这么简单，熟悉 matplotlib 的朋友知道如果需要plot一个数据，我们可以使用 plt.plot(x=, y=)，把x,y的数据作为参数存进去，但是data本来就是一个数据，所以我们可以直接plot。 生成的结果就是下图：

![](./static/pandas测试画图.png)

### 3.9.2 Dataframe 可视化
我们生成一个1000*4 的DataFrame，并对他们累加
```Python
data = pd.DataFrame(
    np.random.randn(1000,4),
    index=np.arange(1000),
    columns=list("ABCD")
    )
data.cumsum()
data.plot()
plt.show()
```
![](./static/pandas测试画图2.png)

这个就是我们刚刚生成的4个column的数据，因为有4组数据，所以4组数据会分别plot出来。plot 可以指定很多参数，具体的用法大家可以自己查一下这里除了plot，我经常会用到还有scatter，这个会显示散点图，首先给大家说一下在 pandas 中有多少种方法
- bar
- hist
- box
- kde
- area
- scatter
- hexbin

但是我们今天不会一一介绍，主要说一下 plot 和 scatter. 因为scatter只有x，y两个属性，我们我们就可以分别给x, y指定数据。
```Python
ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class1')
```

然后我们可以再画一个在同一个ax上面，选择不一样的数据列，不同的 color 和 label
```Python
# 将之下这个 data 画在上一个 ax 上面
data.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
plt.show()
```

下面就是我plot出来的图片
![](./static/pandas测试画图3.png)

这就是我们今天讲的两种呈现方式，一种是线性的方式，一种是散点图。

# 四、附加内容
## 4.1 为什么用 Numpy 还是慢, 你用对了吗?
最近在写代码, 编一个 Python 模拟器, 做 simulation, 好不容易用传说中 Python 里速度最快的计算模块 “Numpy” 的写好了, 结果运行起来, 出奇的慢! 因为一次simulation要一个小时, 要不停测试, 所以自己受不了了。 首先, 我的脑海中的问题, 渐渐浮现出来。

我知道 Pandas 要比 Numpy 慢, 所以我尽量避免用 Pandas。 但是 Numpy (速度怪兽), 为什么还是这么慢?
带有写代码洁癖的我好好给 google 了一番。 第一个出现在我眼前的就是这个文章, Getting the Best Performance out of NumPy。 所以我也将自己从这个文章中学到的诀窍分享给大家, 并补充一些内容。

### 4.1.1、为什么用 Numpy? 
![](./static/numpy图标.jpg)

我们都知道, Python 是慢的, 简单来说, 因为 Python 执行你代码的时候会执行很多复杂的 “check” 功能, 比如当你赋值
```Python
b=1; a=b/0.5
```

这个运算看似简单, 但是在计算机内部, b 首先要从一个整数 integer 转换成浮点数 float, 才能进行后面的 b/0.5, 因为得到的要是一个小数. 还有很多其他的原因和详细说明 (比如 Python 对内存的调用) 在这里能够找到: Why Python is Slow: Looking Under the Hood

提到 Numpy, 它就是一个 Python 的救星. 能把简单好用的 Python 和高性能的 C 语言合并在一起. 当你调用 Numpy 功能的时候, 他其实调用了很多 C 语言而不是纯 Python. 这就是为什么大家都爱用 Numpy 的原因.

### 4.1.2、创建 Numpy Array 的结构
其实 Numpy 就是 C 的逻辑, 创建存储容器 “Array” 的时候是寻找内存上的一连串区域来存放, 而 Python 存放的时候则是不连续的区域, 这使得 Python 在索引这个容器里的数据时不是那么有效率. Numpy 只需要在这块固定的连续区域前后走走就能不费吹灰之力拿到数据. 下图是来自 Why Python is Slow: Looking Under the Hood, 他很好的解释了这一切.
![](./static/Numpy数据读取.png)

在运用 Numpy 的时候, 我们通常不是用一个一维 Array 来存放数据, 而是用二维或者三维的块来存放 (说出了学机器学习的朋友们的心声~)。
![](./static/Numpy二维和三维.png)

因为 Numpy 快速的矩阵相乘运算, 能将乘法运算分配到计算机中的多个核, 让运算并行。 这年头, 我们什么都想 多线程/ 多进程 (再次说出了机器学习同学们的心声~). 这也是 Numpy 为什么受人喜欢的一个原因. 这种并行运算大大加速了运算速度。

那么对于这种天天要用到的2D/3D Array, 我们通常都不会想着他是怎么来的. 因为按照我们正常人的想法, 这矩阵就是矩阵, 没什么深度的东西呀. 不过这可不然! 要不然我也不会写这篇分享了. 重点来了, 不管是1D/2D/3D 的 Array, 从根本上, 它都是一个 1D array!
![](./static/数组如何在内存中存储.png)

这篇 Blog的图显示。 在我们看来的 2D Array, 如果追溯到计算机内存里, 它其实是储存在一个连续空间上的。而对于这个连续空间, 我们如果创建 Array 的方式不同, 在这个连续空间上的排列顺序也有不同。 这将影响之后所有的事情! 我们后面会用 Python 进行运算时间测试。

在 Numpy 中, 创建 2D Array 的默认方式是 “C-type” 以 row 为主在内存中排列, 而如果是 “Fortran” 的方式创建的, 就是以 column 为主在内存中排列。
```Python
col_major = np.zeros((10,10), order='C')    # C-type
row_major = np.zeros((10,10), order='F')    # Fortran
```

### 4.1.3、在 Axis 上的动作 
当你的计算中涉及合并矩阵, 不同形式的矩阵创建方式会给你不同的时间效果。 因为在 Numpy 中的矩阵合并等, 都是发生在一维空间里! 不是我们想象的二维空间中!
```Python
a = np.zeros((200, 200), order='C')
b = np.zeros((200, 200), order='F')
N = 9999

def f1(a):
    for _ in range(N):
        np.concatenate((a, a), axis=0)

def f2(b):
    for _ in range(N):
        np.concatenate((b, b), axis=0)

t0 = time.time()
f1(a)
t1 = time.time()
f2(b)
t2 = time.time()

print((t1-t0)/N)     # 0.000040
print((t2-t1)/N)     # 0.000070
```

从上面的那张图, 可以想到, row 为主的存储方式, 如果在 row 的方向上合并矩阵, 将会更快. 因为只要我们将思维放在 1D array 那, 直接再加一个 row 放在1D array 后面就好了, 所以在上面的测试中, f1 速度要更快. 但是在以 column 为主的系统中, 往 1D array 后面加 row 的规则变复杂了, 消耗的时间也变长. 如果以 axis=1 的方式合并, “F” 方式的 f2 将会比 “C” 方式的 f1 更好.

还有一个要提的事情, 为了图方便, 有时候我会直接使用 np.stack 来代替 np.concatenate, 因为这样可以少写一点代码, 不过使用上面的形式, 通过上面的测试发现是这样. 所以之后为了速度, 我推荐还是尽量使用 np.concatenate.

```Python
np.vstack((a,a))                # 0.000063
np.concatenate((a,a), axis=0)   # 0.000040
```

或者有时候在某个 axis 上进行操作, 比如对上面用 “C-type” 创建的 a 矩阵选点:
```Python
indices = np.random.randint(0, 100, size=10, dtype=np.int32)
a[indices, :]     # 0.000003
a[:, indices]     # 0.000006
```
因为 a 是用 row 为主的形式储存, 所以在 row 上面选数据要比在 column 上选快很多! 对于其他的 axis 的操作, 结果也类似。 所以你现在懂了吧, 看自己要在哪个 axis 上动的手脚多, 然后再创建合适于自己的矩阵形式 (“C-type”/”Fortran”)。

### 4.1.4、copy慢 view快 
在 Numpy 中, 有两个很重要的概念, copy 和 view。 copy 顾名思义, 会将数据 copy 出来存放在内存中另一个地方, 而 view 则是不 copy 数据, 直接取源数据的索引部分. 下图来自 Understanding SettingwithCopyWarning in pandas
![](./static/view和copy.png)

上面说的是什么意思呢? 我们直接看代码。
```Python
a = np.arange(1, 7).reshape((3,2))
a_view = a[:2]
a_copy = a[:2].copy()

a_copy[1,1] = 0
print(a)
"""
[[1 2]
 [3 4]
 [5 6]]
"""

a_view[1,1] = 0
print(a)
"""
[[1 2]
 [3 0]
 [5 6]]
"""
```

简单说, a_view 的东西全部都是 a 的东西, 动 a_view 的任何地方, a 都会被动到, 因为他们在内存中的位置是一模一样的, 本质上就是自己。 而 a_copy 则是将 a copy 了一份, 然后把 a_copy 放在内存中的另外的地方, 这样改变 a_copy, a 是不会被改变的。

那为什么要提这点呢? 因为 view 不会复制东西, 速度快! 我们来测试一下速度。 下面的例子中 a*=2 就是将这个 view 给赋值了, 和 a[:]*=2
一个意思, 从头到尾没有创建新的东西。 而 b = 2*b 中, 我们将 b 赋值给另外一个新建的 b。

```Python
a = np.zeros((1000, 1000))
b = np.zeros((1000, 1000))
N = 9999

def f1(a):
    for _ in range(N):
        a *= 2           # same as a[:] *= 2

def f2(b):
    for _ in range(N):
        b = 2*b

print('%f' % ((t1-t0)/N))     # f1: 0.000837
print('%f' % ((t2-t1)/N))     # f2: 0.001346
```

对于 view 还有一点要提, 你是不是偶尔有时候要把一个矩阵展平, 用到 np.flatten() 或者 np.ravel()。 他俩是不同的! ravel 返回的是一个 view (谢谢知乎上评论的提醒, 官方说如果用 ravel, 需要 copy 的时候才会被 copy , 我想这个时候可能是把 ravel 里面 order 转换的时候, 如 ‘C-type’ -> ‘Fortran’), 而 flatten 返回的总是一个 copy。 现在你知道谁在拖你的后腿了吧! 下面的测试证明, 相比于 flatten, ravel 是神速。

```Python
def f1(a):
    for _ in range(N):
        a.flatten()

def f2(b):
    for _ in range(N):
        b.ravel()

print('%f' % ((t1-t0)/N))    # 0.001059
print('%f' % ((t2-t1)/N))    # 0.000000
```

### 4.1.5、选择数据
选择数据的时候, 我们常会用到 view 或者 copy 的形式。 我们知道了, 如果能用到 view 的, 我们就尽量用 view, 避免 copy 数据。 那什么时候会是 view 呢? 下面举例的都是 view 的方式:
```Python
a_view1 = a[1:2, 3:6]    # 切片 slice
a_view2 = a[:100]        # 同上
a_view3 = a[::2]         # 跳步
a_view4 = a.ravel()      # 上面提到了
...                      # 我只能想到这些, 如果还有请大家在评论里提出
```

那哪些操作我们又会变成 copy 呢?
```Python
a_copy1 = a[[1,4,6], [2,4,6]]   # 用 index 选
a_copy2 = a[[True, True], [False, True]]  # 用 mask
a_copy3 = a[[1,2], :]        # 虽然 1,2 的确连在一起了, 但是他们确实是 copy
a_copy4 = a[a[1,:] != 0, :]  # fancy indexing
a_copy5 = a[np.isnan(a), :]  # fancy indexing
...                          # 我只能想到这些, 如果还有请大家在评论里提出
```

Numpy 给了我们很多很自由的方式选择数据, 这些虽然都很方便, 但是如果你可以尽量避免这些操作, 你的速度可以飞起来。
在上面提到的 blog 里面, 他提到了, 如果你还是喜欢这种 fancy indexing 的形式, 我们也是可以对它加点速的。 那个 blog 中指出了两种方法：

- 1.使用 np.take(), 替代用 index 选数据的方法。
上面提到了如果用index 来选数据, 像 a_copy1 = a[[1,4,6], [2,4,6]], 用 take 在大部分情况中会比这样的 a_copy1 要快。
```Python
a = np.random.rand(1000000, 10)
N = 99
indices = np.random.randint(0, 1000000, size=10000)

def f1(a):
    for _ in range(N):
        _ = np.take(a, indices, axis=0)

def f2(b):
    for _ in range(N):
        _ = b[indices]

print('%f' % ((t1-t0)/N))    # 0.000393
print('%f' % ((t2-t1)/N))    # 0.000569
```

- 2.使用 np.compress(), 替代用 mask 选数据的方法。
上面的 a_copy2 = a[[True, True], [False, True]] 这种就是用 TRUE, FALSE 来选择数据的. 测试如下:
```Python
mask = a[:, 0] < 0.5
def f1(a):
    for _ in range(N):
        _ = np.compress(mask, a, axis=0)

def f2(b):
    for _ in range(N):
        _ = b[mask]

print('%f' % ((t1-t0)/N))    # 0.028109
print('%f' % ((t2-t1)/N))    # 0.031013
```

### 4.1.6、非常有用的 out 参数
不深入了解 numpy 的朋友, 应该会直接忽略很多功能中的这个 out 参数 (之前我从来没用过)。 不过当我深入了解了以后, 发现他非常有用! 比如下面两个其实在功能上是没差的, 不过运算时间上有差, 我觉得可能是 a=a+1 要先转换成 np.add() 这种形式再运算, 所以前者要用更久一点的时间。
```Python
a = a + 1         # 0.035230
a = np.add(a, 1)  # 0.032738
```

如果是上面那样, 我们就会触发之前提到的 copy 原则, 这两个被赋值的 a, 都是原来 a 的一个 copy, 并不是 a 的 view. 但是在功能里面有一个 out 参数, 让我们不必要重新创建一个 a. 所以下面两个是一样的功能, 都不会创建另一个 copy. 不过可能是上面提到的那个原因, 这里的运算时间也有差。
```Python
a += 1                 # 0.011219
np.add(a, 1, out=a)    # 0.008843
```
带有 out 的 numpy 功能都在这里:Universal functions. 所以只要是已经存在了一个 placeholder (比如 a), 我们就没有必要去再创建一个, 用 out 方便又有效。

### 4.1.7、给数据一个名字
我喜欢用 pandas, 因为 pandas 能让你给数据命名, 用名字来做 index。 在数据类型很多的时候, 名字总是比 index 好记太多了, 也好用太多了。 但是 pandas 的确比 numpy 慢。 好在我们还是有途径可以实现用名字来索引。 这就是 structured array。 下面 a/b 的结构是一样的, 只是一个是 numpy 一个是 pandas。
```Python
a = np.zeros(3, dtype=[('foo', np.int32), ('bar', np.float16)])
b = pd.DataFrame(np.zeros((3, 2), dtype=np.int32), columns=['foo', 'bar'])
b['bar'] = b['bar'].astype(np.float16)

"""
# a
array([(0,  0.), (0,  0.), (0,  0.)],
      dtype=[('foo', '<i4'), ('bar', '<f2')])

# b
   foo  bar
0    0  0.0
1    0  0.0
2    0  0.0
"""

def f1(a):
    for _ in range(N):
        a['bar'] *= a['foo']

def f2(b):
    for _ in range(N):
        b['bar'] *= b['foo']

print('%f' % ((t1-t0)/N))    # 0.000003
print('%f' % ((t2-t1)/N))    # 0.000508
```

可以看出来, numpy 明显比 pandas 快很多。 如果需要使用到不同数据形式, numpy 也是可以胜任的, 并且在还保持了快速的计算速度。 至于 pandas 为什么比 numpy 慢, 因为 pandas data 里面还有很多七七八八的数据, 记录着这个 data 的种种其他的特征。 这里还有更全面的对比: Numpy Vs Pandas Performance Comparison

如果大家还有其他的小技巧或者是速度大比拼, 欢迎在下面讨论. (一切为了速度~)

最后, 如果你对机器学习感兴趣, 这里有很多厉害的短片形式机器学习方法介绍和很多机器学习的 Python 实践教程, 让你可以用业余时间秒懂机器学习.


