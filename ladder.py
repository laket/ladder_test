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

        act = b1 + w1z*z_n + w1u*u + w1zu*z_n*u
        z_h = b0 + w0z*z_n + w0u*u + w0zu*z_n*u + ws * tf.sigmoid(act)
    
    return z_h

def _normalize(v, mu, var, epsilon=1e-10):
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
            
            return z, z, None, None


        W = var_wd("W", [Dp, D])
        # [N, Dn] = [N, Dp] * [Dp, Dn]
        z_pre = tf.matmul(h_pre, W)
        Semi supervised learning with Ladder Networkの追試をやってたんだけど、95%までしかでない。(原論文は99%でる)
        # [Dn]
        mu, var = tf.nn.moments(z_pre, axes=[0])
        
        if noise_std is not None:
            noise = tf.random_normal([N, D]) * noise_std
            z = _normalize(z_pre, mu, var) + noise
        else:
            z = _normalize(z_pre, mu, var)

        if self.act == tf.nn.relu:
            beta  = var_wd("beta", [D], tf.constant_initializer(0.0))
            activation = z + beta
        elif self.act == tf.identity:
            beta  = var_wd("beta", [D], tf.constant_initializer(0.0))
            gamma = var_wd("gamma", [D], tf.constant_initializer(1.0))
            activation = gamma * (z+beta)
        else:
            raise ValueError("unknown activation function")

        h = self.act(activation)

        return z, h, mu, var

    def forward(self, h_pre, reuse=True):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            # h_c is not used

            with tf.name_scope("Encoder_clean"):
                self.z_c, self.h_c, self.mu_c, self.var_c = self._forward(h_pre, noise_std=None)
                activation_summary(self.h_c)
            return self.h_c

    def noise_forward(self, h_pre, reuse=True):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
                
            # h_n is not used but in last layer
            with tf.name_scope("Encoder_noisy"):
                self.z_n, self.h_n, _, _ = self._forward(h_pre, self.noise_std)
            return self.h_n

    def backward(self, z_next):
        """
        define backward path

        Args:
          z_next: activation of next layer. [N, Dn]
        """
        Dn = z_next.get_shape()[1]
        D = self.D

        with tf.variable_scope(self.name) as scope:
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

    def reconstruct_cost(self):
        """
        calculate reconstruction cost at this layer.

        In the artcile backward path aims to reconstruct pre batch-normalized activation.
        But output of the first layer is not batch-normalized.
        So this compares no-batch-normalized value with reconstructed value at the first layer.
        (this is not clear in articles for me)
        """
        if self.is_first:
            # [N, D]
            target = self.z_h
        else:
            # [N,D] = ([N,D] - D) / D
            target = _normalize(self.z_h, self.mu_c, self.var_c)            

        # scale factor comes from the original paper page 8.
        return 2*tf.nn.l2_loss(self.z_c - target) / (self.D*self.batch_size)

class LadderNetwork(object):
    def __init__(self, act=tf.nn.relu):
        """
        Args:
          act: activation function
        """
        self.act = act
        self.layers = None

        """
        hidden layer sizes from the original article.
        There are other two layers as input and output.
        """
        self.layer_sizes = [1000, 500, 250, 250, 250]

        # noise level from the original article
        # 0.3 is the best at N=100 labels
        # 0.2 is the best at N=1000 labels
        self.noise_std = 0.3

        # weight of each layer for reconstruction error.
        self.loss_lambda = [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    def forward(self, x, is_first=False):
        layer_sizes = self.layer_sizes
        output_size = 10

        N, D = x.get_shape().as_list()
        noise_std = self.noise_std

        layers = []

        # define first layer
        first_layer = Layer("input", D, batch_size=N, noise_std=noise_std, is_first=True)
        h_c = first_layer.forward(x)
        h_n = first_layer.noise_forward(x)
        layers.append(first_layer)

        # define hidden layers
        for idx, layer_size in enumerate(layer_sizes):
            layer = Layer("layer{}".format(idx), layer_size, batch_size=N, noise_std=noise_std)
            h_c = layer.forward(h_c, reuse=(not is_first))
            h_n = layer.noise_forward(h_n)

            layers.append(layer)

        """
        activation of last layer is identity (for classification)
        Because outputting logits is better than softmax for stability of loss funcation
        """
        last_layer = Layer("last", output_size, batch_size=N, noise_std=noise_std, is_last=True, act=tf.identity)
        logits_c = last_layer.forward(h_c, reuse=(not is_first))
        logits_n = last_layer.noise_forward(h_n)
        layers.append(last_layer)

        self.layers = layers

        return logits_c, logits_n

    def backward(self):
        """
        Add loss function around backward pass
        """
        if self.layers is None:
            raise ValueError("call forward before backward!")

        last_layer = self.layers[-1]
        #z_next = last_layer.h_n
        z_next = tf.nn.softmax(last_layer.h_n)
        
        for layer in reversed(self.layers):
            z_next = layer.backward(z_next)

    def total_loss(self, x, y, semi_x):
        """
        Args:
          x: supervised x [N, D]
          y: supervised y [N, M]. onehot vectors
          semi_x: semi-supervised x [N2, D]
        """
        _, noisy_logits = self.forward(x, is_first=True)
        supervised_loss = self.supervised_loss(y, noisy_logits)

        self.forward(semi_x)
        self.backward()
        unsupervised_loss = self.unsupervised_loss() 

        loss = tf.add(supervised_loss, unsupervised_loss, name="total_loss")
        return loss

    def supervised_loss(self, labels, logits):
        """
        Args:
          labels: one hot true labels.
          logits: logits of prediction.
        """
        #weight_decay = 0.0004
        #weight_decay = 0.0
        #weight_loss = tf.add_n(tf.get_collection('L2_LOSS')) * weight_decay
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        loss = tf.reduce_mean(cross_entropy, name="cross_entropy")

        tf.add_to_collection("summary_loss", loss)

        return loss

    def unsupervised_loss(self):
        """
        calculate unsupervised loss (reconstruction loss).
        This must be called after 'backward'.
        """
        costs = []
        
        for lamb, layer in zip(self.loss_lambda, self.layers):
            costs.append(lamb * layer.reconstruct_cost())

        semi_cost = tf.reduce_sum(costs, name="ReconsCost")
        tf.add_to_collection("summary_loss", semi_cost)
        
        return semi_cost
