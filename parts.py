#!/usr/bin/python

import tensorflow as tf

def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  #tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def var_wd(name, shape, initializer=None, wd=1.0):
    """

    Args:
      name: name of variable
      shape: shape of variable
      initializer: initializer
      wd: relative weight decay factor with other variables.
          If wd is None, the variables is escaped from weight decay.
    Returns:
      Tensor of Variable
    """
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    var = tf.get_variable(name, shape, initializer=initializer)

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='l2_loss')
        tf.add_to_collection('L2_LOSS', weight_decay)

    return var
