# -*- coding:utf-8 -*-

"""
Define Ladder Network

Based on
-Deconstructing the Ladder Network Architecture.
 I refers section2 of this article.
 Variables names of this program comes from it.
-Semi-Supervised Learning with Ladder Networks
 I refers Algorithm1 of this article.
 https://arxiv.org/abs/1507.02672

Abbreviate
N : minibatch size
D : dimension of a feature
n : noise
c : clean
h : hat

Variables
z_n : outputs of noizy encoder
z_c : outputs of clean encoder
z   : outputs of encoders
z_h : outputs of decoder
y   : target variable

"""

import numpy as np
import tensorflow as tf

from parts import var_wd, activation_summary

def vanilla_combinator(z_n, u):
    """
    Args:
      z_n : lateral connection [N, D]
      u   : vertical connection [N, D]
    Returns:
      z_h : output of decoder [N, D]
    """
    D = u.get_shape()[1]

    with tf.variable_scope("combinator"):
        v0 = lambda name: var_wd(name, [D], tf.constant_initializer(0.0))
        v1 = lambda name: var_wd(name, [D], tf.constant_initializer(1.0))
        
        w0z, w1z = v1("w_0z"), v1("w_1z")
        w0u, w1u = v0("w_0u"), v0("w_1u")
        w0zu, w1zu, b0, b1 = v0("w_0zu"), v0("w_1zu"), v0("b0"), v0("b1")
        ws = v1("w_sigma")

        act = b1 + w1z*z_n + w1z*u + w1zu*z_n*u
        z_h = b0 + w0z*z_n + w0u*u + w0zu*z_n*u + ws * tf.sigmoid(act)
    
    return z_h

def _normalize(v, mu, var, epsilon=0.001):
    inv = tf.rsqrt(var + epsilon)
    
    return (v-mu)*inv

class Layer(object):
    def __init__(self, name, D, batch_size, noise_std, is_first=False, is_last=False, act=tf.nn.relu, comb=vanilla_combinator):
        """
        Args:
          name: name of layer
          D   : dimension of hidden unit
          batch_size : batch_size of input (int)
          noise_std : standard deviation of noise
          act : activation function
          comb : combinator function for backward
        """
        
        self.name = name
        self.D = D
        self.batch_size = batch_size
        # combinator function
        self.comb = comb
        self.noise_std = noise_std
        self.is_first = is_first
        self.is_last = is_last
        self.act = act

    def _forward(self, h_pre, noise_std=None):
        """
        define forward path.
        store self.z
        
        Args:
          h_pre: post activation of the below layer. [N,Dp]
          noise_std : standard deviation of noise (if None, don't add noise to outputs)

        Returns:
          h    : post activation of this layer
        """
        Dp = h_pre.get_shape()[1]
        D = self.D
        N = self.batch_size

        if self.is_first:
            if noise_std is not None:
                noise = tf.random_normal([N, D]) * noise_std
                z = h_pre + noise
            else:
                z = h_pre
            
            return h_pre, h_pre

        with tf.name_scope("encoder"):
            W = var_wd("W", [Dp, D])
            # [N, Dn] = [N, Dp] * [Dp, Dn]
            z_pre = tf.matmul(h_pre, W)
        
            # [Dn]
            mu, var = tf.nn.moments(z_pre, axes=[0])

            if noise_std is not None:
                noise = tf.random_normal([N, D]) * noise_std
                z = _normalize(z_pre, mu, var) + noise
            else:
                z = _normalize(z_pre, mu, var)
                
            beta  = var_wd("beta", [D], tf.constant_initializer(0.0))
            gamma = var_wd("gamma", [D], tf.constant_initializer(1.0))

            if self.is_last and noise_std is None:
                self.logits = gamma * (z + beta)
            h = self.act(gamma * (z + beta))

        return z, h

    def forward(self, h_pre):
        with tf.variable_scope(self.name):
            self.z_c, h = self._forward(h_pre, noise_std=None)
            activation_summary(h)
            return h

    def noise_forward(self, h_pre):
        with tf.variable_scope(self.name) as scope:
            scope.reuse_variables()            
            self.z_n, h = self._forward(h_pre, self.noise_std)
            return h

    def backward(self, z_next):
        """
        define backward path

        Args:
          z_next: activation of next layer. [N, Dn]
        """
        Dn = tf.get_shape(z_next)[1]
        D = self.D

        with tf.variable_scope("decoder"):
            if self.is_last:
                u_pre = z_next
            else:
                V = var_wd("V", [Dn, D],)
                u_pre = tf.matmul(z_next, V)

            mu, var = tf.nn.moments(u_pre, axes=[0])
            u = _normalize(u_pre, mu, var)
        
            self.z_h = self.comb(self.z_n, u)

        return self.z_h

class LadderNetwork(object):
    def init(self, act=tf.nn.relu):
        """
        Args:
          act: activation function
        """
        self.act = act
        self.layers = None
    
    def forward(self, x):
        layer_sizes = [1000, 500, 250, 250, 250]
        output_size = 10

        N, D = x.get_shape().as_list()
        noise_std = 0.3

        layers = []

        # define first layer
        first_layer = Layer("input", D, batch_size=N, noise_std=noise_std, is_first=True)
        h_c = first_layer.forward(x)
        h_n = first_layer.noise_forward(x)
        layers.append(first_layer)

        # define hidden layers
        for idx, layer_size in enumerate(layer_sizes):
            layer = Layer("layer{}".format(idx), layer_size, batch_size=N, noise_std=0.01)
            h_c = layer.forward(h_c)
            h_n = layer.noise_forward(h_n)

            layers.append(layer)

        # last layer is softmax (for classification)
        last_layer = Layer("last", output_size, batch_size=N, noise_std=noise_std, is_last=True, act=tf.nn.softmax)
        y_c = last_layer.forward(h_c)
        y_n = last_layer.noise_forward(h_n)
        layers.append(last_layer)

        self.layers = layers

        #return y_c, y_n
        return last_layer.logits

    def backward(self, y_n):
        """
        Add loss function around backward pass
        """
        if self.layers is None:
            raise ValueError("call forward before backward!")

        z_next = y_n
        for idx_layer, layer in reversed(self.layers):
            z_next = layer.backward(z_next)

    def supervised_loss(self, labels, logits):
        """
        Args:
          labels: one hot true labels.
          logits: logits of prediction.
        """
        #weight_decay = 0.0004
        weight_decay = 0.0
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        weight_loss = tf.add_n(tf.get_collection('L2_LOSS')) * weight_decay

        total_loss = loss + weight_loss
        
        tf.add_to_collection("cross_entropy", loss)
        tf.add_to_collection("total_loss", total_loss)

        return total_loss
            
