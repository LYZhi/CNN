import numpy as np

import optim


class Solver(object):
  """
  A Solver encapsulates all the logic necessary for training classification
  models. The Solver performs stochastic gradient descent using different
  update rules defined in optim.py.

  The solver accepts both training and validataion data and labels so it can
  periodically check classification accuracy on both training and validation
  data to watch out for overfitting.

  To train a model, you will first construct a Solver instance, passing the
  model, dataset, and various optoins (learning rate, batch size, etc) to the
  constructor. You will then call the train() method to run the optimization
  procedure and train the model.
  
  After the train() method returns, model.params will contain the parameters
  that performed best on the validation set over the course of training.
  In addition, the instance variable solver.loss_history will contain a list
  of all losses encountered during training and the instance variables
  solver.train_acc_history and solver.val_acc_history will be lists containing
  the accuracies of the model on the training and validation set at each epoch.
  
  Example usage might look something like this:
  
  data = {
    'X_train': # training data
    'y_train': # training labels
    'X_val': # validation data
    'X_train': # validation labels
  }
  model = MyAwesomeModel(hidden_size=100, reg=10)
  solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
  solver.train()


  A Solver works on a model object that must conform to the following API:

  - model.params must be a dictionary mapping string parameter names to numpy
    arrays containing parameter values.

  - model.loss(X, y) must be a function that computes training-time loss and
    gradients, and test-time classification scores, with the following inputs
    and outputs:

    Inputs:
    - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].

    Returns:
    If y is None, run a test-time forward pass and return:
    - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].

    If y is not None, run a training time forward and backward pass and return
    a tuple of:
    - loss: Scalar giving the loss
    - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
  """

  def __init__(self, model, data, **kwargs):
    """
    Construct a new Solver instance.
    
    Required arguments:
    - model: A model object conforming to the API described above
    - data: A dictionary of training and validation data with the following:
      'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
      'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
      'y_train': Array of shape (N_train,) giving labels for training images
      'y_val': Array of shape (N_val,) giving labels for validation images
      
    Optional arguments:
    - update_rule: A string giving the name of an update rule in optim.py.
      Default is 'sgd'.
    - optim_config: A dictionary containing hyperparameters that will be
      passed to the chosen update rule. Each update rule requires different
      hyperparameters (see optim.py) but all update rules require a
      'learning_rate' parameter so that should always be present.
    - lr_decay: A scalar for learning rate decay; after each epoch the learning
      rate is multiplied by this value.
    - batch_size: Size of minibatches used to compute loss and gradient during
      training.
    - num_epochs: The number of epochs to run for during training.
    - print_every: Integer; training losses will be printed every print_every
      iterations.
    - verbose: Boolean; if set to false then no output will be printed during
      training.
    """
    # 第一步：从data中获得训练数据和验证集数据
    self.model = model
    self.X_train = data['X_train']
    self.y_train = data['y_train']
    self.X_val = data['X_val']
    self.y_val = data['y_val']
    
    # Unpack keyword arguments
    # 第二步：获得传入的参数，学习率衰减值，训练的batch_size, epoch的大小，学习率和momentum， 梯度下降的方式
    """
    solver = Solver(model, data,                
                lr_decay=0.95,                
                print_every=10, num_epochs=5, batch_size=2, 
                update_rule='sgd_momentum',
                optim_config={'learning_rate': 5e-4, 'momentum': 0.9})

    """
    # 如果没有update_rule，就返回sgd
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.lr_decay = kwargs.pop('lr_decay', 1.0)
    self.batch_size = kwargs.pop('batch_size', 2)
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    # 如果存在未知的输入则报错
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)

    # Make sure the update rule exists, then replace the string
    # name with the actual function
    # 如果optim中不存在梯度下降的方式就报错
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)

    # 将optim中的函数功能赋予函数名self.update_rule
    self.update_rule = getattr(optim, self.update_rule)

    # 第三步：进行部分初始化操作
    self._reset()


  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """
    # Set up some variables for book-keeping
    # 迭代epoch的次数
    self.epoch = 0
    # 最好的验证集的准确率
    self.best_val_acc = 0
    self.best_params = {}
    # 损失值的list
    self.loss_history = []
    # 准确率的list
    self.train_acc_history = []
    self.val_acc_history = []

    # Make a deep copy of the optim_config for each parameter
    # 每个dw和db对应的学习率和momentum
    self.optim_configs = {}
    # 建立每个参数对应的学习率和momentum
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.items()}
      self.optim_configs[p] = d

  # 计算loss和进行参数更新
  def _step(self):
    """
    Make a single gradient update. This is called by train() and should not
    be called manually.
    """
    # Make a minibatch of training data
    # 获得当前样本的个数 500
    num_train = self.X_train.shape[0]
    # 从测试集500个中随机抽取2个样本，用于进行参数的更新
    batch_mask = np.random.choice(num_train, self.batch_size)
    # 获取到样本的数据和标签
    X_batch = self.X_train[batch_mask]
    y_batch = self.y_train[batch_mask]

    # Compute loss and gradient
    # 计算损失值和梯度方向
    loss, grads = self.model.loss(X_batch, y_batch)
    self.loss_history.append(loss)

    # Perform a parameter update
    # 对每个参数进行循环
    for p, w in self.model.params.items():
      dw = grads[p]
      # 获得当前的学习率和momentum, 以及后续加入的v
      config = self.optim_configs[p]
      # 将w，dw, config传入到动量梯度算法，进行参数更新
      next_w, next_config = self.update_rule(w, dw, config)
      # 将更新后的参数替换成模型中的参数
      self.model.params[p] = next_w
      # 将更新后的config替代字典中的config
      self.optim_configs[p] = next_config


  # 进行准确率的计算
  def check_accuracy(self, X, y, num_samples=None, batch_size=2):
    """
    Check accuracy of the model on the provided data.
    
    Inputs:
    - X: Array of data, of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,)
    - num_samples: If not None, subsample the data and only test the model
      on num_samples datapoints.
    - batch_size: Split X and y into batches of this size to avoid using too
      much memory.
      
    Returns:
    - acc: Scalar giving the fraction of instances that were correctly
      classified by the model.
    """
    
    # Maybe subsample the data
    # num_sample表示使用多少个数据计算准确率
    N = X.shape[0]
    if num_samples is not None and N > num_samples:
      # 随机从N个样本中，抽取num_sample个样本 -- 4个样本
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

    # Compute predictions in batches
    num_batches = (int)(N / batch_size)
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    # batch进行交叉验证，每次只进行部分的结果预测
    for i in range(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      # 不传入y，获得scores得分
      scores = self.model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
    # 将数据进行横向排列
    y_pred = np.hstack(y_pred)
    # 计算结果的平均值
    acc = np.mean(y_pred == y)

    return acc


  def train(self):
    """
    Run optimization to train the model.
    """

    # 第四步：使用样本数和batch_size，即epoch_num，构造迭代的次数
    num_train = self.X_train.shape[0]
    # shap 500  batch_size 2
    iterations_per_epoch = max((int)(num_train / self.batch_size), 1)
    # iterations_per_epoch 250
    # num_epochs 5
    # num_iterations 500
    num_iterations = self.num_epochs * iterations_per_epoch

    for t in range(num_iterations):
      # 第五步：循环，计算损失值和梯度值，并使用sgd_momentum进行参数更新
      self._step()

      # Maybe print training loss
      # 第六步：每一个print_every打印损失值
      if self.verbose and t % self.print_every == 0:
        print ('(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1]))

      # At the end of every epoch, increment the epoch counter and decay the
      # learning rate.
      # 第七步：每一个循环进行一次学习率的下降
      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end:
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]['learning_rate'] *= self.lr_decay

      # Check train and val accuracy on the first iteration, the last
      # iteration, and at the end of each epoch.
      # 检查值的准确性
      first_it = (t == 0)
      last_it = (t == num_iterations + 1)

      # 第八步：开始或者结束，或者每一个epoch计算准确率，同时获得验证集最好的参数
      if first_it or last_it or epoch_end:
        train_acc = self.check_accuracy(self.X_train, self.y_train,num_samples=4)
        val_acc = self.check_accuracy(self.X_val, self.y_val,num_samples=4)
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        if self.verbose:
          print ('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                 self.epoch, self.num_epochs, train_acc, val_acc))

        # Keep track of the best model
        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          self.best_params = {}
          for k, v in self.model.params.items():
            self.best_params[k] = v.copy()

    # At the end of training swap the best params into the model
    # 将验证集最好的参数赋予给当前的模型参数
    self.model.params = self.best_params

