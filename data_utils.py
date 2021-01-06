import pickle as pickle
import numpy as np
import os
#from scipy.misc import imread


# 读取到当前数据集的数据与标签
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  # 第三步：打开文件，进行数据的读取
  with open(filename, 'rb') as f:
    # 文件的读取
    datadict = pickle.load(f,encoding='latin1')
    # 获得数据与标签
    """
    一个测试集中有一万张图片，data的数据结构为10000×3072
    数据的每一行都存储一个32×32的图片，其中的每1024个数据分别代表了红绿蓝三通道
    图像按照行顺序存储，所以数组的前32个数据是图像的第一行的红色通道值
    """
    X = datadict['data']
    # 一万张图片所对应的标签
    Y = datadict['labels']
    # 进行数据的维度重构
    """
    reshape将(10000, 3072)转换为（10000×3×32×32）结构
    transpose 将每个RGB的值放在一起转换为（32×32×3）
    """

    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    # 将标签转换为np.array格式
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  # 第二步：创建列表，用于文件数据的保存
  xs = []
  ys = []

  # 获取的是第一个数据集
  for b in range(1,2):
    # 拼接成数据集的文件目录
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    # 获取到data和labels
    X, Y = load_CIFAR_batch(f)
    # 使用list进行保存
    xs.append(X)
    ys.append(Y)

  # 将数据串联拼接
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  # 获得测试数据
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


# 加载数据集
def get_CIFAR10_data(num_training=200, num_validation=50, num_test=200):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data

    # 输入文件地址
    cifar10_dir = '/home/zhi/cifar-10-batches-py/'
    # 获得训练数据和测试数据
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    print (X_train.shape)
    # Subsample the data
    # 对数据进行二次采样
    # 第四步，创建一个mask索引，用于生成50个验证集val数据， 500个训练数据和500个测试数据
    # 从训练样本中选验证集[500.550)
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    # 从训练样本中选训练集 [0,500）
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    # 从测试样本中选测试集 [0,500)
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    """
    第五步：对图像进行均值化操作——
    自然图像其实是一种平稳的数据分布即图像的每一维都服从相同的分布。
    所以通过减去数据对应维度的统计平均值，
    来消除公共的部分，以凸显个体之间的特征和差异。
    
    """
    # 输出矩阵是一行，按列求平均值,得到训练集的均值，
    # 将每个训练集均减去公共部分，
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # 第六步：将图片的维度进行转换 返回深拷贝
    # 转换为（500,3,32,32）
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # 第七步：创建一个字典返回数据
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
    
"""
def load_tiny_imagenet(path, dtype=np.float32):
  
  Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
  TinyImageNet-200 have the same directory structure, so this can be used
  to load any of them.

  Inputs:
  - path: String giving path to the directory to load.
  - dtype: numpy datatype used to load the data.

  Returns: A tuple of
  - class_names: A list where class_names[i] is a list of strings giving the
    WordNet names for class i in the loaded dataset.
  - X_train: (N_tr, 3, 64, 64) array of training images
  - y_train: (N_tr,) array of training labels
  - X_val: (N_val, 3, 64, 64) array of validation images
  - y_val: (N_val,) array of validation labels
  - X_test: (N_test, 3, 64, 64) array of testing images.
  - y_test: (N_test,) array of test labels; if test labels are not available
    (such as in student code) then y_test will be None.
  
  # First load wnids
  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # Map wnids to integer labels
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # Use words.txt to get names for each class
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.iteritems():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]

  # Next load training data.
  X_train = []
  y_train = []
  for i, wnid in enumerate(wnids):
    if (i + 1) % 20 == 0:
      print 'loading training data for synset %d / %d' % (i + 1, len(wnids))
    # To figure out the filenames we need to open the boxes file
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]
    num_images = len(filenames)
    
    X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
    y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
    for j, img_file in enumerate(filenames):
      img_file = os.path.join(path, 'train', wnid, 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        ## grayscale file
        img.shape = (64, 64, 1)
      X_train_block[j] = img.transpose(2, 0, 1)
    X_train.append(X_train_block)
    y_train.append(y_train_block)
      
  # We need to concatenate all training data
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)
  
  # Next load validation data
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []
    val_wnids = []
    for line in f:
      img_file, wnid = line.split('\t')[:2]
      img_files.append(img_file)
      val_wnids.append(wnid)
    num_val = len(img_files)
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
    X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
      img_file = os.path.join(path, 'val', 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        img.shape = (64, 64, 1)
      X_val[i] = img.transpose(2, 0, 1)

  # Next load test images
  # Students won't have test labels, so we need to iterate over files in the
  # images directory.
  img_files = os.listdir(os.path.join(path, 'test', 'images'))
  X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
  for i, img_file in enumerate(img_files):
    img_file = os.path.join(path, 'test', 'images', img_file)
    img = imread(img_file)
    if img.ndim == 2:
      img.shape = (64, 64, 1)
    X_test[i] = img.transpose(2, 0, 1)

  y_test = None
  y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
  if os.path.isfile(y_test_file):
    with open(y_test_file, 'r') as f:
      img_file_to_wnid = {}
      for line in f:
        line = line.split('\t')
        img_file_to_wnid[line[0]] = line[1]
    y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    y_test = np.array(y_test)
  
  return class_names, X_train, y_train, X_val, y_val, X_test, y_test

"""
def load_models(models_dir):
  """
  Load saved models from disk. This will attempt to unpickle all files in a
  directory; any files that give errors on unpickling (such as README.txt) will
  be skipped.

  Inputs:
  - models_dir: String giving the path to a directory containing model files.
    Each model file is a pickled dictionary with a 'model' field.

  Returns:
  A dictionary mapping model file names to models.
  """
  models = {}
  for model_file in os.listdir(models_dir):
    with open(os.path.join(models_dir, model_file), 'rb') as f:
      try:
        models[model_file] = pickle.load(f)['model']
      except pickle.UnpicklingError:
        continue
  return models
