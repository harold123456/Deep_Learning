# TensorFlow
[TOC]
## 简介

### 张量值
- 阶为维数
- 形状为整数元祖

### 构建计算图(tf.Graph)
- 操作（简称“op”）：图的节点。操作描述了消耗和生成张量的计算。
- 张量：图的边。它们代表将流经图的值。大多数 TensorFlow 函数会返回tf.Tensors。

### 运行计算图(tf.Session)
```
sess = tf.Session()
print(sess.run(tf.Tensors))
```

### 供给
- 占位符(参数化接受外部输入)，类似于函数参数
```
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
```
- feed_dict(给占位符提供具体的值)
```
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
```
  - feed_dict也可以给tf.constant传递新值
```
sess = tf.InteractiveSession()

a = tf.constant(5)
b  = a.eval(feed_dict={a:2})
print(b)   # 输出2
```

### 数据集(将数据流式传输到模型)

### 层
- 创建层
```python
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```
- ==初始化层==
```
init = tf.global_variables_initializer()
sess.run(init)
```
- 执行层
```
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
>
[[-3.41378999]
 [-9.14999008]]
```
- 层函数的快捷方式
  - 区别是单次调用中创建和运行层
```
y = tf.layers.dense(x, units=1)
```

### 损失
```
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

print(sess.run(loss))
```

### 训练
```
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```
- 循环
```
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)
```

### 完整程序
```
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
```

## Tensors
### 主要类型
- tf.Variable
- tf.constant
- tf.placeholder
- tf.SparseTensor

### 更改数据类型
```
# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
```

## Variables
### 创建变量
- 只提供名称和形状
```
my_variable = tf.get_variable("my_variable", [1, 2, 3])

my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer())

other_variable = tf.get_variable("other_variable", dtype=tf.int32,
  initializer=tf.constant([23, 42]))   # 此时不指定变量形状
```


### 初始化变量
```
session.run(tf.global_variables_initializer())
# Now all variables are initialized.
```
- 用其它变量的值初始化一个新的变量时，使用其它变量的initialized_value()属性
```
# Create a variable with a random value.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
```

### 保存变量
```
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print "Model saved in file: ", save_path
```

### 恢复变量
```
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print "Model restored."
  # Do some work with the model
  ...
```
## 卷积神经网络
### 权重初始化
```
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) # 截断正态分布，只取2个标准差内的
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
```

### 卷积和池化
```
def conv2d(x,W):
    # input.shape = [batch, in_height, in_width, in_channels]
    # filter.shape = [filter_height, filter_width, in_channels, out_channels]
    # Must have `strides[0] = strides[3] = 1`.
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #此处的SAME是为了使奇数行列的最后一列也进行pool
```

### 第一层卷积(第二层类似)
```
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

### 全连接层
```
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

### Dropout层
- 为什么取rate=0.5，由下式知，当 n=0.5*m时，值最大
```math
C_m^n
```
```
rate = tf.placeholder(dtype=tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,rate=rate)
```

### softmax层
```
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```

### 训练和评估模型
```
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))   # 交叉熵
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  # 训练方法
correct_predict = tf.equal(tf.argmax(y_, axis=1),tf.argmax(y_conv, axis=1))   # 输出为bool，所以要转换成数值
accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32))

with tf.Session() as sess:    # 用这个可以使用op.run()和t.eval()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # op.run()和t.eval()可以替代sess.run(),op指operate，t指Tensor
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], rate: 0.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], rate: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, rate: 0.0}))
```

### 完整代码
```
import input_data
import tensorflow as tf

# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷积和池化
def conv2d(x,W):
    # input.shape = [batch, in_height, in_width, in_channels]
    # filter.shape = [filter_height, filter_width, in_channels, out_channels]
    # Must have `strides[0] = strides[3] = 1`.
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # sess = tf.InteractiveSession()
    x = tf.placeholder('float', shape=[None, 784])
    y_ = tf.placeholder('float', shape=[None, 10])
    # 第一层卷积
    '''卷积在每个5x5的patch中算出32个特征。
    卷积的权重张量形状是[5, 5, 1, 32]，
    前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 
    而对于每一个输出通道都有一个对应的偏置量。'''
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # dense层
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout层
    rate = tf.placeholder(dtype=tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,rate=rate)

    # softmax层
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 训练和评估模型
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    correct_predict = tf.equal(tf.argmax(y_, axis=1),tf.argmax(y_conv, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                # op.run()和t.eval()可以替代sess.run(),op指operate，t指Tensor
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], rate: 0.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], rate: 0.5})

        print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, rate: 0.0}))

```





