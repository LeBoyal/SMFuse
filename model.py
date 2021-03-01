# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm,
  tf_ms_ssim,
  tf_ssim,
  Smooth_l1_loss
)

import time
import os
import matplotlib.pyplot as plt
import cv2 

import numpy as np
import tensorflow as tf

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=84,
               label_size=84,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        self.images_ir = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
        self.labels_ir = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir')
    with tf.name_scope('VI_input'):

        self.images_vi = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')
        self.labels_vi = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi')
    with tf.name_scope('Mask_input'):
        self.images_mask = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim],
                                          name='images_mask')
        self.labels_mask = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim],
                                          name='labels_mask')

    # with tf.name_scope('IR_inputtu'):
    #     self.images_irtu = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_irtu')
    #     self.labels_irtu = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_irtu')
    #
    # with tf.name_scope('VI_inputtu'):
    #     self.images_vitu = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vitu')
    #     self.labels_vitu = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vitu')





    with tf.name_scope('input'):
        self.input_image_ir =self.labels_ir
        self.input_image_vi =self.labels_vi
        self.input_image_mask =self.labels_mask
        # self.input_image_irtu = self.labels_irtu
        # self.input_image_vitu = self.labels_vitu



    with tf.name_scope('fusion'): 
        # self.fusion_image=self.fusion_model(self.input_image_ir,self.input_image_vi)
        self.fusion_map, self.fusion_mask  = self.fusion_model(self.input_image_ir,self.input_image_vi)
        self.fusion_image = self.fusion_mask * self.labels_vi + (1 - self.fusion_mask) * self.labels_ir

    with tf.name_scope('grad_bin'):
        self.Image_vi_grad = gradient(self.labels_vi)
        self.Image_ir_grad = gradient(self.labels_ir)
        self.Image_fused_grad = gradient(self.fusion_image)
        self.Image_max_grad = tf.round((self.Image_vi_grad + self.Image_ir_grad) // (tf.abs(self.Image_vi_grad + self.Image_ir_grad) + 0.0000000001)) * tf.maximum(tf.abs(self.Image_vi_grad), tf.abs(self.Image_ir_grad))
        # self.Image_vi_score=tf.reduce_mean(tf.square(self.Image_vi_grad))
        # self.Image_ir_score=tf.reduce_mean(tf.square(self.Image_ir_grad))

               
        # self.Image_vi_ir_grad_bin = tf.concat([self.Image_vi_grad, self.Image_ir_grad], 3)
        # self.Image_fused_grad_bin = tf.concat([self.Image_fused_grad, self.Image_fused_grad], 3)
    


    with tf.name_scope('image'):
        tf.summary.image('input_ir',tf.expand_dims(self.labels_ir[1,:,:,:],0))  
        tf.summary.image('input_vi',tf.expand_dims(self.labels_vi[1,:,:,:],0))
        # tf.summary.image('input_irtu',tf.expand_dims(self.labels_irtu[1,:,:,:],0))
        # tf.summary.image('input_vitu',tf.expand_dims(self.labels_vitu[1,:,:,:],0))
        tf.summary.image('mask', tf.expand_dims(self.labels_mask[1, :, :, :], 0))
        tf.summary.image('fusion_map',tf.expand_dims(self.fusion_map[1,:,:,:],0))
        tf.summary.image('fusion_mask', tf.expand_dims(self.fusion_mask[1, :, :, :], 0))
        tf.summary.image('fusion_image', tf.expand_dims(self.fusion_image[1, :, :, :], 0))
        # tf.summary.image('Image_vi_grad',tf.expand_dims(self.Image_vi_grad[1,:,:,:],0))
        # tf.summary.image('Image_ir_grad',tf.expand_dims(self.Image_ir_grad[1,:,:,:],0))
        # tf.summary.image('Image_max_grad', tf.expand_dims(self.Image_max_grad[1, :, :, :], 0))
        # tf.summary.image('Image_vi_ir_grad_bin',tf.expand_dims(self.Image_vi_ir_grad_bin[1,:,:,:],0))
        # tf.summary.image('Image_fused_grad_bin',tf.expand_dims(self.Image_fused_grad_bin[1,:,:,:],0))
        


          
    # with tf.name_scope('d_loss'):
    #     pos=self.discriminator(self.labels_mask,reuse=False)
    #     neg=self.discriminator(self.fusion_image,reuse=True,update_collection='NO_OPS')
    #     pos_loss=tf.reduce_mean(tf.square(pos-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
    #     neg_loss=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))
    #     self.d_loss=neg_loss+pos_loss
    #     tf.summary.scalar('loss_d',self.d_loss)


        
    with tf.name_scope('g_loss'):
        # self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        # tf.summary.scalar('g_loss_1',self.g_loss_1)

        # self.g_loss_int =  tf.reduce_mean((self.Image_vi_weight-tf.minimum(self.Image_vi_weight,self.Image_ir_weight))*tf.square(self.fusion_image - self.labels_vi)) +\
        #                    tf.reduce_mean((self.Image_ir_weight-tf.minimum(self.Image_vi_weight,self.Image_ir_weight))*tf.square(self.fusion_image - self.labels_ir))
        
        self.g_loss_grau = tf.reduce_mean(tf.square(self.Image_fused_grad - self.Image_max_grad))
        
        # self.g_loss_int = tf.reduce_mean(tf.square(self.fusion_map - self.labels_mask))
        self.g_loss_int = Smooth_l1_loss(self.labels_mask, self.fusion_map)

        self.g_loss = self.g_loss_grau + 10*self.g_loss_int


        # 5e-3



        # tf.summary.scalar('self.g_loss_int', self.g_loss_int)
        tf.summary.scalar('self.g_loss_grau', self.g_loss_grau)
        tf.summary.scalar('self.g_loss_int', self.g_loss_int)
        # tf.summary.scalar('g_loss_2',self.g_loss_2)
        # self.g_loss_total=self.g_loss_1+ self.g_loss_2
        tf.summary.scalar('loss_g',self.g_loss)
        
    self.saver = tf.train.Saver(max_to_keep=50)
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,"Train_ir")
      input_setup(self.sess,config,"Train_vi")
      # input_setup(self.sess, config,"Train_irtu")
      # input_setup(self.sess,config,"Train_vitu")
      input_setup(self.sess, config, "mask")
    else:
      nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir")
      nx_vi,ny_vi=input_setup(self.sess, config,"Test_vi")
      nx_mask, ny_mask = input_setup(self.sess, config, "mask")

    if config.is_train:     
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir","train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi","train.h5")
      # data_dir_irtu = os.path.join('./{}'.format(config.checkpoint_dir), "Train_irtu","train.h5")
      # data_dir_vitu = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vitu","train.h5")
      data_dir_mask = os.path.join('./{}'.format(config.checkpoint_dir), "mask", "train.h5")
    else:
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir),"Test_ir", "test.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir),"Test_vi", "test.h5")
      data_dir_mask = os.path.join('./{}'.format(config.checkpoint_dir), "mask", "test.h5")

    train_data_ir, train_label_ir = read_data(data_dir_ir)
    train_data_vi, train_label_vi = read_data(data_dir_vi)
    # train_data_irtu, train_label_irtu = read_data(data_dir_irtu)
    # train_data_vitu, train_label_vitu = read_data(data_dir_vitu)
    train_data_mask, train_label_mask = read_data(data_dir_mask)

    t_vars = tf.trainable_variables()
    # self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    # print(self.d_vars)
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print(self.g_vars)

    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss,var_list=self.g_vars)
        # self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)

    self.summary_op = tf.summary.merge_all()

    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()



    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_ir) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          # batch_images_irtu = train_data_irtu[idx*config.batch_size : (idx+1)*config.batch_size]
          # batch_labels_irtu = train_label_irtu[idx*config.batch_size : (idx+1)*config.batch_size]
          # batch_images_vitu = train_data_vitu[idx*config.batch_size : (idx+1)*config.batch_size]
          # batch_labels_vitu = train_label_vitu[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_mask = train_data_mask[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_mask = train_label_mask[idx*config.batch_size : (idx+1)*config.batch_size]


          counter += 1
          # for i in range(2):
          #   _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi,self.images_mask: batch_images_mask, self.labels_vi: batch_labels_vi,self.labels_ir:batch_labels_ir,self.labels_mask:batch_labels_mask})

          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss,self.summary_op], feed_dict={self.images_ir: batch_images_ir,self.images_vi: batch_images_vi,self.images_mask: batch_images_mask, self.labels_ir: batch_labels_ir, self.labels_vi: batch_labels_vi, self.labels_mask:batch_labels_mask})

          self.train_writer.add_summary(summary_str,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_g))


        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_mask.eval(feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir, self.images_vi: train_data_vi, self.labels_vi: train_label_vi,self.images_mask: train_data_mask,self.labels_mask: train_label_mask})
      result=result*127.5+127.5
      result = merge(result, [nx_ir, ny_ir])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def fusion_model(self,img_ir,img_vi):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 1, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [16], initializer=tf.constant_initializer(0.0))
            img_ir = tf.pad(img_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv1_ir = tf.nn.conv2d(img_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer1_vi'):
            weights = tf.get_variable("w1_vi", [3, 3, 1, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1_vi", [16], initializer=tf.constant_initializer(0.0))
            img_vi = tf.pad(img_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv1_vi = tf.nn.conv2d(img_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv1_vi = lrelu(conv1_vi)
            print("conv1_vi:", conv1_vi.shape)

        ####################  Layer2  ###########################
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", [3, 3, 16, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [16], initializer=tf.constant_initializer(0.0))
            conv1_ir_ = tf.pad(conv1_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv2_ir = tf.nn.conv2d(conv1_ir_, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv2_ir = lrelu(conv2_ir)

        with tf.variable_scope('layer2_vi'):
            weights = tf.get_variable("w2_vi", [3, 3, 16, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2_vi", [16], initializer=tf.constant_initializer(0.0))
            conv1_vi_ = tf.pad(conv1_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv2_vi = tf.nn.conv2d(conv1_vi_, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv2_vi = lrelu(conv2_vi)
            print("conv2_vi:", conv2_vi.shape)

        ####################  Layer3  ###########################
        conv_12_ir = tf.concat([conv1_ir, conv2_ir], axis=-1)
        conv_12_vi = tf.concat([conv1_vi, conv2_vi], axis=-1)
        print("conv_12_vi:", conv_12_vi.shape)

        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", [3, 3, 32, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b3", [16], initializer=tf.constant_initializer(0.0))
            conv_12_ir = tf.pad(conv_12_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv3_ir = tf.nn.conv2d(conv_12_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer3_vi'):
            weights = tf.get_variable("w3_vi", [3, 3, 32, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b3_vi", [16], initializer=tf.constant_initializer(0.0))
            conv_12_vi = tf.pad(conv_12_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv3_vi = tf.nn.conv2d(conv_12_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv3_vi = lrelu(conv3_vi)
            print("conv3_vi:", conv3_vi.shape)

        ####################  Layer4  ###########################
        conv_123_ir = tf.concat([conv1_ir, conv2_ir, conv3_ir], axis=-1)
        conv_123_vi = tf.concat([conv1_vi, conv2_vi, conv3_vi], axis=-1)
        print("conv_123_vi:", conv_123_vi.shape)

        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", [3, 3, 48, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b4", [16], initializer=tf.constant_initializer(0.0))
            conv_123_ir = tf.pad(conv_123_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv4_ir = tf.nn.conv2d(conv_123_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer4_vi'):
            weights = tf.get_variable("w4_vi", [3, 3, 48, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b4_vi", [16], initializer=tf.constant_initializer(0.0))
            conv_123_vi = tf.pad(conv_123_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv4_vi = tf.nn.conv2d(conv_123_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv4_vi = lrelu(conv4_vi)
            print("conv4_vi:", conv4_vi.shape)

        conv_ir_vi = tf.concat([conv1_ir, conv1_vi, conv2_ir, conv2_vi, conv3_ir, conv3_vi, conv4_ir, conv4_vi],
                               axis=-1)
        print("conv_ir_vi:", conv_ir_vi.shape)

        with tf.variable_scope('layer5'):
            weights = tf.get_variable("w5", [1, 1, 128, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b5", [1], initializer=tf.constant_initializer(0.0))
            conv5_ir = tf.nn.conv2d(conv_ir_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv5_ir = lrelu(conv5_ir)
            print("conv5_ir:", conv5_ir.shape)

        with tf.variable_scope('layer6'):
            weights = tf.get_variable("w6", [1, 1, 64, 32],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b6", [1], initializer=tf.constant_initializer(0.0))
            conv6_ir = tf.nn.conv2d(conv5_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv6_ir = lrelu(conv6_ir)
            print("conv6_ir:", conv6_ir.shape)

        with tf.variable_scope('layer7'):
            weights = tf.get_variable("w7", [1, 1, 32, 1],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b7", [1], initializer=tf.constant_initializer(0.0))
            conv7_ir = tf.nn.conv2d(conv6_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            map = tf.nn.tanh(conv7_ir) / 2 + 0.5
            # conv6_ir = tf.nn.sigmoid(conv6_ir)
            print("map:", map.shape)
            mask = binary(map - 0.5)
            mask = mask / 2 + 0.5
    return map, mask

  # def fusion_model(self,img_ir):
  #   with tf.variable_scope('fusion_model'):
  #       with tf.variable_scope('layer1'):
  #           weights = tf.get_variable("w1", [3, 3, 1, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias = tf.get_variable("b1", [16], initializer=tf.constant_initializer(0.0))
  #           img_ir = tf.pad(img_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
  #           conv1_ir = tf.nn.conv2d(img_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
  #           conv1_ir = lrelu(conv1_ir)
  #
  #
  #       ####################  Layer2  ###########################
  #       with tf.variable_scope('layer2'):
  #           weights = tf.get_variable("w2", [3, 3, 16, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias = tf.get_variable("b2", [16], initializer=tf.constant_initializer(0.0))
  #           conv1_ir_ = tf.pad(conv1_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
  #           conv2_ir = tf.nn.conv2d(conv1_ir_, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
  #           conv2_ir = lrelu(conv2_ir)
  #
  #       ####################  Layer3  ###########################
  #       conv_12_ir = tf.concat([conv1_ir, conv2_ir], axis=-1)
  #
  #       with tf.variable_scope('layer3'):
  #           weights = tf.get_variable("w3", [3, 3, 32, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias = tf.get_variable("b3", [16], initializer=tf.constant_initializer(0.0))
  #           conv_12_ir = tf.pad(conv_12_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
  #           conv3_ir = tf.nn.conv2d(conv_12_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
  #           conv3_ir = lrelu(conv3_ir)
  #
  #
  #       ####################  Layer4  ###########################
  #       conv_123_ir = tf.concat([conv1_ir, conv2_ir, conv3_ir], axis=-1)
  #
  #
  #
  #       with tf.variable_scope('layer4'):
  #           weights = tf.get_variable("w4", [3, 3, 48, 16], initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias = tf.get_variable("b4", [16], initializer=tf.constant_initializer(0.0))
  #           conv_123_ir = tf.pad(conv_123_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
  #           conv4_ir = tf.nn.conv2d(conv_123_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
  #           conv4_ir = lrelu(conv4_ir)
  #
  #
  #       conv_ir_vi = tf.concat([conv1_ir, conv2_ir, conv3_ir,  conv4_ir],
  #                              axis=-1)
  #       print("conv_ir_vi:", conv_ir_vi.shape)
  #
  #       with tf.variable_scope('layer5'):
  #           weights = tf.get_variable("w5", [1, 1, 64, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias = tf.get_variable("b5", [1], initializer=tf.constant_initializer(0.0))
  #           conv5_ir = tf.nn.conv2d(conv_ir_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
  #           conv5_ir = lrelu(conv5_ir)
  #           print("conv5_ir:", conv5_ir.shape)
  #
  #       with tf.variable_scope('layer6'):
  #           weights = tf.get_variable("w6", [1, 1, 32, 1],
  #                                     initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias = tf.get_variable("b6", [1], initializer=tf.constant_initializer(0.0))
  #           conv6_ir = tf.nn.conv2d(conv5_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
  #           map = tf.nn.tanh(conv6_ir) / 2 + 0.5
  #           # conv6_ir = tf.nn.sigmoid(conv6_ir)
  #           print("map:", map.shape)
  #           mask = binary(map - 0.5)
  #           mask = mask / 2 + 0.5
  #
  #   return map, mask





    
    
    
  # def discriminator(self,img,reuse,update_collection=None):
  #   with tf.variable_scope('discriminator',reuse=reuse):
  #       print(img.shape)
  #       with tf.variable_scope('layer_1'):
  #           weights=tf.get_variable("w_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_1",[32],initializer=tf.constant_initializer(0.0))
  #           conv1_vi=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
  #           conv1_vi = lrelu(conv1_vi)
  #           #print(conv1_vi.shape)
  #       with tf.variable_scope('layer_2'):
  #           weights=tf.get_variable("w_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_2",[64],initializer=tf.constant_initializer(0.0))
  #           conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv2_vi = lrelu(conv2_vi)
  #           #print(conv2_vi.shape)
  #       with tf.variable_scope('layer_3'):
  #           weights=tf.get_variable("w_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_3",[128],initializer=tf.constant_initializer(0.0))
  #           conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv3_vi=lrelu(conv3_vi)
  #           #print(conv3_vi.shape)
  #       with tf.variable_scope('layer_4'):
  #           weights=tf.get_variable("w_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_4",[256],initializer=tf.constant_initializer(0.0))
  #           conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_vi, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
  #           conv4_vi=lrelu(conv4_vi)
  #           [B,H,W,C]=conv4_vi.get_shape().as_list()
  #
  #           conv4_vi = tf.reshape(conv4_vi,[self.batch_size,H*H*256])
  #       with tf.variable_scope('line_5'):
  #           weights=tf.get_variable("w_5",[H*H*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
  #           bias=tf.get_variable("b_5",[1],initializer=tf.constant_initializer(0.0))
  #           line_5=tf.matmul(conv4_vi, weights) + bias
  #   return line_5

  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False

def binary(input):
    x = input
    # with tf.get_default_graph().gradient_override_map({"Sign": 'QuantizeGrad'}):
    x = tf.sign(x)
    return x


