#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
training Ladder Network
"""

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import mnist_input
import ladder

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dir_log', './log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('dir_parameter', './parameter',
                           """Directory where to write parameters""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

########  optimization parameters #############
tf.app.flags.DEFINE_integer('flat_steps', 100,
                            """the number of steps using initial learning rate""")
tf.app.flags.DEFINE_integer('decay_steps', 50,
                            """the number of steps until linearly decreasing to 0""")
tf.app.flags.DEFINE_float('lr', 0.002,
                          "initial learning rate")


def add_loss_summaries():
  """
  Add summaries for losses
  This depends on collection("cross_entropy") and collection("total_loss")
  
  Return:
    op for generating moving averages of losses.
  """
  raw_loss = tf.get_collection("cross_entropy")[0]
  total_loss = tf.get_collection("total_loss")[0]
    
  loss_averages = tf.train.ExponentialMovingAverage(0.95, name='avg')
  loss_averages_op = loss_averages.apply([raw_loss, total_loss])

  for name, l in [("cross_entropy", raw_loss), ("total_loss", total_loss)]:
    tf.scalar_summary(name +' (raw)', l)
    tf.scalar_summary(name, loss_averages.average(l))

  return loss_averages_op


def get_train_op(total_loss, global_step):
  """
  gets train operator
  Create an optimizer and apply to all trainable variables. 
  
  Args:
    total_loss: Total loss
    global_step: Integer Variable counting the number of training steps

  Returns:
    op for training updates variables
  """
  """
  lr = tf.train.exponential_decay(FLAGS.lr,
                                  global_step,
                                  FLAGS.decay_steps,
                                  FLAGS.decay,
                                  staircase=True)
  """
  lr = tf.select(
    tf.less(global_step,FLAGS.flat_steps),
    FLAGS.lr,
    FLAGS.lr * (1 - tf.cast(tf.truediv((global_step - FLAGS.flat_steps), FLAGS.decay_steps), tf.float32) )
  )

  

  tf.scalar_summary('learning_rate', lr)

  loss_average_op = add_loss_summaries()
  
  # Compute gradients.
  with tf.control_dependencies([total_loss, loss_average_op]):
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

  return train_op

def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    images, labels = mnist_input.train_input()

    network = ladder.LadderNetwork()

    logits = network.forward(images)
    total_loss = network.supervised_loss(labels, logits)
    
    train_op = get_train_op(total_loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.trainable_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)    
    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.dir_log, sess.graph)

    max_steps = FLAGS.flat_steps + FLAGS.decay_steps
    for step in xrange(max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, total_loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0 or (step + 1) == max_steps:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 10 == 0 or (step + 1) == max_steps:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 100 == 0 or (step + 1) == max_steps:
        checkpoint_path = os.path.join(FLAGS.dir_parameter, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  mnist_input.init()
  
  if tf.gfile.Exists(FLAGS.dir_log):
    tf.gfile.DeleteRecursively(FLAGS.dir_log)
  tf.gfile.MakeDirs(FLAGS.dir_log)
    
  if tf.gfile.Exists(FLAGS.dir_parameter):
    tf.gfile.DeleteRecursively(FLAGS.dir_parameter)

  tf.gfile.MakeDirs(FLAGS.dir_parameter)

  train()


if __name__ == '__main__':
  tf.app.run()

