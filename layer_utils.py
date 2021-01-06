from layers import *


# 线性传播和池化层的前向传播，即全连接层的前向传播
def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache

# 线性传播和池化层的反向传播，即全连接层的反向传播
def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_naive(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_naive(da, conv_cache)
  return dx, dw, db

# 卷积层，激活层，池化层的前向传播
def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """

  # 卷积层的前向传播--y = w * x + b
  # conv_param 卷集参数
  # a1, conv_cache1 = conv_forward_naive(x, w, b, conv_param)
  a, conv_cache = conv_forward_naive(x, w, b, conv_param)




  """
  由 y = w * x + b 可知，如果不用激活函数，
  每个网络层的输出都是一种线性输出，
  而我们所处的现实场景，其实更多的是各种非线性的分布。
  这也说明了激活函数的作用是将线性分布转化为非线性分布，
  能更逼近我们的真实场景。
  s --非线性分布
  """
  # relu层的前向传播， np.maxmuim(0, x) 小于零的值使用零表示
  s, relu_cache = relu_forward(a)
  """
  减小输入矩阵的大小（只是宽和高，而不是深度），提取主要特征
  pool_param --池化层
  """
  # pool层的前向传播，对卷积部分的图像求出最大值，作为pool池化后的大小
  out, pool_cache = max_pool_forward_naive(s, pool_param)
  """
  损失函数，通过梯度计算dw，db，Relu激活函数逆变换，反池化，反全连接
  """
  # 将各个输入组合成一个cache，用于反向传播
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache

# pool,relu, conv的反向传播
def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  # 获得三个层的输入参数
  conv_cache, relu_cache, pool_cache = cache
  # 进行池化层的反向传播，构造最大值的[[false, false], [false, True]]列表，最大值部分不变，其他部位使用0值填充
  ds = max_pool_backward_naive(dout, pool_cache)
  # 进行relu层的反向传播，dout[x<0] = 0, 将输入小于0的dout置为0
  da = relu_backward(ds, relu_cache)
  # 卷积层的反向传播，对dx, dw, db进行反向传播，dx[i, :, j*s] += dout * w[f], dw[f] += windows * dout, db[f] += dout
  dx, dw, db = conv_backward_naive(da, conv_cache)
  return dx, dw, db

