#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
validate Ladder Network
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

tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                            """How often to run the eval.""")


def restore_model(saver, sess):
  ckpt = tf.train.get_checkpoint_state(FLAGS.dir_parameter)
  if ckpt and ckpt.model_checkpoint_path:
    # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
  else:
    print('No checkpoint file found')
    return None
  
  return global_step

def eval_once(summary_writer, top_k_op):
  saver = tf.train.Saver(tf.trainable_variables())
  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()

  # Start running operations on the Graph.

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01) 
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess.run(init)
  global_step = restore_model(saver, sess)

  if global_step is None:
    return

  # Start the queue runners.
  coord = tf.train.Coordinator()    
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  true_count = 0  # Counts the number of correct predictions.
  total_sample_count = mnist_input.VALIDATION_SIZE
  step = 0
  num_iter = total_sample_count / FLAGS.batch_size
    
  for i in range(num_iter):
    predictions = sess.run(top_k_op)
    true_count += np.sum(predictions)
    step += 1

  # Compute precision @ 1.
  precision = true_count / float(total_sample_count)
  print('%s: step %d precision @ 1 = %.3f' % (datetime.now(), int(global_step), precision))

  summary = tf.Summary()
  summary.value.add(tag='Precision @ 1', simple_value=precision)
  summary_writer.add_summary(summary, global_step)            


  coord.request_stop()
  coord.join(threads, stop_grace_period_secs=10)
    

def evaluate():
  with tf.Graph().as_default() as g, tf.device("/cpu:0"):
    FLAGS.batch_size = 100
    images, labels = mnist_input.validate_input()

    network = ladder.LadderNetwork()

    logits = network.forward(images)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    summary_writer = tf.train.SummaryWriter(FLAGS.dir_log, g)

    while True:
      eval_once(summary_writer, top_k_op)
      time.sleep(FLAGS.eval_interval_secs)

        
def main(argv=None):  # pylint: disable=unused-argument
  mnist_input.init()
  evaluate()

if __name__ == '__main__':
  tf.app.run()

