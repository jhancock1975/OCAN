#!/usr/bin/env python
"""
    Original Author's info
    Please see https://github.com/PanpanZheng/OCAN/blob/master/oc_gan.py

    Author: Panpan Zheng
    Date created:  2/15/2018
    Python Version: 2.7

    Bibtex reference for paper:

@article{zheng2018one,
  title={One-Class Adversarial Nets for Fraud Detection},
  author={Zheng, Panpan and Yuan, Shuhan and Wu, Xintao and Li, Jun and Lu, Aidong},
  journal={arXiv preprint arXiv:1803.01798},
  year={2018}
}

  This code is known to work with tensorflow 1.14 and Python 3.6
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import os

from bg_utils import pull_away_loss, one_hot, xavier_init, sample_shuffle_spv, sample_shuffle_uspv, sample_Z, draw_trend
from bg_dataset import load_data, load_data_unbal
import sys

class OneClassGan(object):
  """
  based on https://github.com/PanpanZheng/OCAN ocgan.py
  """

  def __init__(self, mb_size, n_maj_class, pretrain_size, n_round, max_instances=None, pos_label=1, neg_label=0  ):
    """
    Parameters
    ----------
    mb_size: int
      minibatch size for training
    n_maj_class: int
      number of instances of majority class to use
      for training
    pretrain_size: int
      number of instances to use for pretraining
    max_instances: int
      maximum number of instances of the minority and majority classes to use
      for trainng
   n_round: int
      number of training epochs to run
    """ 
    self._mb_size = mb_size
    self._n_maj_class = n_maj_class
    self._pretrain_size = pretrain_size
    self._pos_label = pos_label
    self._neg_label = neg_label

  def fit(X, y):
      """
      fit one-class generative adversarial network classifier
      this code was modified from oc_gan.py
      https://github.com/PanpanZheng/OCAN/blob/master/oc_gan.py
      
      Parameters
      ----------
      X: list-like
        two dimensional array of features
      y: list-like
        ground-truth values for supervised learning
      """
      dim_input = X.shape[1]
      D_dim = [dim_input, 100, 50, 2]
      G_dim = [50, 100, dim_input]
      Z_dim = G_dim[0]
      
      
      # define placeholders for labeled-data, unlabeled-data, noise-data and target-data.
      
      X_oc = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_input])
      Z = tf.compat.v1.placeholder(tf.float32, shape=[None, Z_dim])
      X_tar = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_input])
      # X_val = tf.placeholder(tf.float32, shape=[None, dim_input])
      
      
      # declare weights and biases of discriminator.
      
      D_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))
      D_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))
      
      D_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))
      D_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))
      
      D_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))
      D_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))
      
      theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
      
      
      
      # declare weights and biases of generator.
      
      G_W1 = tf.Variable(xavier_init([G_dim[0], G_dim[1]]))
      G_b1 = tf.Variable(tf.zeros(shape=[G_dim[1]]))
      
      G_W2 = tf.Variable(xavier_init([G_dim[1], G_dim[2]]))
      G_b2 = tf.Variable(tf.zeros(shape=[G_dim[2]]))
      
      theta_G = [G_W1, G_W2, G_b1, G_b2]
      
      
      # declare weights and biases of pre-train net for density estimation.
      
      T_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))
      T_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))
      
      T_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))
      T_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))
      
      T_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))
      T_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))
      
      theta_T = [T_W1, T_W2, T_W3, T_b1, T_b2, T_b3]
      
      
      def generator(z):
          G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
          G_logit = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
          return G_logit
      
      
      def discriminator(x):
          D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
          D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
          D_logit = tf.matmul(D_h2, D_W3) + D_b3
          D_prob = tf.nn.softmax(D_logit)
          return D_prob, D_logit, D_h2
      
      
      # pre-train net for density estimation.
      
      def discriminator_tar(x):
          T_h1 = tf.nn.relu(tf.matmul(x, T_W1) + T_b1)
          T_h2 = tf.nn.relu(tf.matmul(T_h1, T_W2) + T_b2)
          T_logit = tf.matmul(T_h2, T_W3) + T_b3
          T_prob = tf.nn.softmax(T_logit)
          return T_prob, T_logit, T_h2
      
      
      D_prob_real, D_logit_real, D_h2_real = discriminator(X_oc)
      
      G_sample = generator(Z)
      D_prob_gen, D_logit_gen, D_h2_gen = discriminator(G_sample)
      
      D_prob_tar, D_logit_tar, D_h2_tar = discriminator_tar(X_tar)
      D_prob_tar_gen, D_logit_tar_gen, D_h2_tar_gen = discriminator_tar(G_sample)
      # D_prob_val, _, D_h1_val = discriminator(X_val)
      
      
      
      
      # disc. loss
      y_real= tf.compat.v1.placeholder(tf.int32, shape=[None, D_dim[3]])
      y_gen = tf.compat.v1.placeholder(tf.int32, shape=[None, D_dim[3]])
      
      D_loss_real = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_real,labels=tf.stop_gradient(y_real)))
      D_loss_gen = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_gen, labels=tf.stop_gradient(y_gen)))
      
      ent_real_loss = -tf.reduce_mean(
                              input_tensor=tf.reduce_sum(
                                  input_tensor=tf.multiply(D_prob_real, tf.math.log(D_prob_real)), axis=1
                              )
                          )
      
      ent_gen_loss = -tf.reduce_mean(
                              input_tensor=tf.reduce_sum(
                                  input_tensor=tf.multiply(D_prob_gen, tf.math.log(D_prob_gen)), axis=1
                              )
                          )
      
      D_loss = D_loss_real + D_loss_gen + 1.85 * ent_real_loss
      
      
      # gene. loss
      pt_loss = pull_away_loss(D_h2_tar_gen)
      
      y_tar= tf.compat.v1.placeholder(tf.int32, shape=[None, D_dim[3]])
      T_loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_tar, labels=tf.stop_gradient(y_tar)))
      tar_thrld = tf.divide(tf.reduce_max(input_tensor=D_prob_tar_gen[:,-1]) +
                            tf.reduce_min(input_tensor=D_prob_tar_gen[:,-1]), 2)
      
      # tar_thrld = tf.reduce_mean(D_prob_tar_gen[:,-1])
      
      
      indicator = tf.sign(
                    tf.subtract(D_prob_tar_gen[:,-1],
                                tar_thrld))
      condition = tf.greater(tf.zeros_like(indicator), indicator)
      mask_tar = tf.compat.v1.where(condition, tf.zeros_like(indicator), indicator)
      G_ent_loss = tf.reduce_mean(input_tensor=tf.multiply(tf.math.log(D_prob_tar_gen[:,-1]), mask_tar))
      # G_ent_loss = tf.reduce_mean(tf.log(D_prob_tar_gen[:,-1]))
      
      fm_loss = tf.reduce_mean(
                  input_tensor=tf.sqrt(
                      tf.reduce_sum(
                          input_tensor=tf.square(D_logit_real - D_logit_gen), axis=1
                          )
                      )
                  )
      
      G_loss = pt_loss + G_ent_loss + fm_loss
      
      D_solver = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=theta_D)
      G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
      T_solver = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(T_loss, var_list=theta_T)
      
      
      idx = np.random.randint(X.shape[0], size=self._n_maj_class)
      x_benign = X[y == self._neg_label][idx]
      x_vandal = X[y == self._pos_labl]
     
      x_benign = sample_shuffle_uspv(x_benign)
      x_vandal = sample_shuffle_uspv(x_vandal)
      
      if self._max_instances:
          x_benign = x_benign[0:self._max_instances]
          x_vandal = x_vandal[0:self._max_instances]
          x_pre = x_benign[0:self._pretrain_size]
      else:
          x_pre = x_benign[0:self._pretrain_size]
      
      y_pre = np.zeros(len(x_pre))
      y_pre = one_hot(y_pre, 2)
      
      x_train = x_pre
      
      y_real_mb = one_hot(np.zeros(mb_size), 2)
      y_fake_mb = one_hot(np.ones(mb_size), 2)
     
      
      sess = tf.compat.v1.Session()
      self._sess = sess
      sess.run(tf.compat.v1.global_variables_initializer())
      
      # pre-training for target distribution
      
      _ = sess.run(T_solver,
                   feed_dict={
                      X_tar:x_pre,
                      y_tar:y_pre
                      })
      
      q = int(np.divide(len(x_train), mb_size))
      
      # n_epoch = 1
      #
      # while n_epoch:
      
      d_ben_pro, d_fake_pro, fm_loss_coll = list(), list(), list()
      f1_score  = list()
      d_val_pro = list()
      
      
      for n_epoch in range(n_round):
      
          X_mb_oc = sample_shuffle_uspv(x_train)
      
          for n_batch in range(q):
      
              _, D_loss_curr, ent_real_curr = sess.run([D_solver, D_loss, ent_real_loss],
                                                feed_dict={
                                                           X_oc: X_mb_oc[n_batch*mb_size:(n_batch+1)*mb_size],
                                                           Z: sample_Z(mb_size, Z_dim),
                                                           y_real: y_real_mb,
                                                           y_gen: y_fake_mb
                                                           })
      
              _, G_loss_curr, fm_loss_curr = sess.run([G_solver, G_loss, fm_loss],
              # _, G_loss_curr, fm_loss_, kld_ = sess.run([G_solver, G_loss, fm_loss, pt_loss + G_ent_loss],
                                                 feed_dict={Z: sample_Z(mb_size, Z_dim),
                                                            X_oc: X_mb_oc[n_batch*mb_size:(n_batch+1)*mb_size],
                                                            })
      
      self._sess = sess
      self._D_prob_real = D_prob_real
      self._D_logit_real = D_logit_real
      self._X_oc = X_oc
      self._trained = True
          # print conf_mat

    class NotTrainedError(Exception):
      """ Thrown when predict_proba called before fit is called.
      """      
      pass

    def  predict_proba(X):
      """
      return class probabilities for instances X
      
      Parameters
      ----------
      X: list-like
        two dimensional array of classifier input values
      """
      if not self._trained:
        raise  NotTrainedError("OneClassGan is not trained yet.  Please call the fit() function first")
      else:
        prob, _ = self._sess.run([self._D_prob_real, self._D_logit_real], feed_dict={self._X_oc: x_test})
        return prob

      


if __name__ == "__main__":
  exit(0)
