"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
from six.moves import urllib

import numpy as np
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

tf.app.flags.DEFINE_string("dir_data", "./data", "data directory")

"""
batch size is the same between labeled smaple and unlabeled sample.
This comes from "add_train_params" in
https://github.com/CuriousAI/ladder/blob/master/run.py

"""
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")

FLAGS = tf.app.flags.FLAGS
#VALIDATION_SIZE = 10000
VALIDATION_SIZE = 59900
NUM_CLASSES = 10

train_data_filename = None
train_labels_filename = None
test_data_filename = None
test_labels_filename = None
iter_per_epoch = None

def init():
  """
  do initialization of this module.
  If we don't have files in local, this start to download.
  """
  global train_data_filename
  global train_labels_filename
  global test_data_filename
  global test_labels_filename
  global iter_per_epoch

  print ("start data download")
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
  test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
  test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')  

  print ("finish data download")

  iter_per_epoch = VALIDATION_SIZE / FLAGS.batch_size  

def train_input():
  train_data = extract_data(train_data_filename, 60000)
  train_labels = extract_labels(train_labels_filename, 60000)

  data = train_data[VALIDATION_SIZE:, ...]
  labels = train_labels[VALIDATION_SIZE:]

  image, label = tf.train.slice_input_producer((data, labels), shuffle=True, capacity=4096, name="label_input")
  return make_batch(image, label)


def fast_train_input():
  """
  cache preprocessed result
  """
  train_data = extract_data(train_data_filename, 60000)
  train_labels = extract_labels(train_labels_filename, 60000)

  data = train_data[VALIDATION_SIZE:, ...]
  labels = train_labels[VALIDATION_SIZE:]

  value = np.reshape(data, [len(data), -1]) / 255 - 0.5
  images = tf.constant(value, dtype=tf.float32)
  label_vector = tf.one_hot(labels, NUM_CLASSES)
  
  return images, label_vector


def unlabeled_train_input():
  train_data = extract_data(train_data_filename, 60000)
  data = train_data[:VALIDATION_SIZE, ...]

  queue_image = tf.train.input_producer(data, shuffle=True, capacity=4096, name="unlabel_input")
  image = queue_image.dequeue()

  return make_batch(image)


def validate_input():
  train_data = extract_data(train_data_filename, 60000)
  train_labels = extract_labels(train_labels_filename, 60000)

  data = train_data[:VALIDATION_SIZE, ...]
  labels = train_labels[:VALIDATION_SIZE]

  image, label = tf.train.slice_input_producer((data, labels), shuffle=False, capacity=4096, name="validate_input")
  return make_batch(image, label, need_onehot=False)

def test_input():
  data = extract_data(test_data_filename, 60000)
  labels = extract_labels(test_labels_filename, 60000)

  image, label = tf.train.slice_input_producer((data, labels), shuffle=False)
  return make_batch(image, label)
  

def preprocess(src):
  """
  do common image preprocess among train and test.
  [0,255] -> [-0.5, 0.5]
  
  Args:
    src: Tensor [h, w, channel]
  Returns:
    dest: Tensor [h*w*channel]
  """
  src = tf.reshape(src,[-1])
  
  return (src / 255) - 0.5

def make_batch(image, label=None, need_onehot=True):
  """
  preprocess image and labe. Than make batch.
  This doesn't shuffle.

  Args:
    image: Tensor of image data
    label: Tensor of label data. If None, label is not contained in batch
    need_onehot: whether to make onehot vector for label
  """
  preprocessed = preprocess(image)

  if need_onehot and label is not None:
    label_vector = tf.one_hot(label, NUM_CLASSES)
  else:
    label_vector = label

  num_preprocess_threads = 8
  #num_preprocess_threads = 2
  capacity = 4096

  if label is not None:
    images, label_batch = tf.train.batch(
      [preprocessed, label_vector],
      batch_size=FLAGS.batch_size,
      num_threads=num_preprocess_threads,
      capacity=capacity,
      name="label_batch"
    )

    return images, label_batch
  
  else:
    images = tf.train.batch(
      [preprocessed],
      batch_size=FLAGS.batch_size,
      num_threads=num_preprocess_threads,
      capacity=capacity,
      name="unlabel_batch"
    )

    return images


  


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  WORK_DIRECTORY = FLAGS.dir_data
  
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
    
  return filepath

def extract_data(filename, num_images):
  """
  Extract the images into a 4D tensor.

  Args:
    filename: image filename
    num_images: the numer of image in file

  Returns:
    data : image ranges [0,255] Tensor(image index, y, x, channels)
  """
  IMAGE_SIZE = 28
  NUM_CHANNELS = 1

  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    
    return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels
  
