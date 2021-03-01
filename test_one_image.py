# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2




def imread1(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, flatten=False, mode='RGB').astype(np.float)

def imread2(path, is_grayscale=False):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, flatten=False, mode='RGB').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img_ir,img_vi):
    with tf.variable_scope('fusion_model'):
    
####################  Layer1  ###########################
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias = tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            img_ir = tf.pad(img_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv1_ir = tf.nn.conv2d(img_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer1_vi'):
            weights = tf.get_variable("w1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/w1_vi')))
            bias = tf.get_variable("b1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/b1_vi')))
            img_vi = tf.pad(img_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv1_vi = tf.nn.conv2d(img_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv1_vi = lrelu(conv1_vi)
            print("conv1_vi:", conv1_vi.shape)
        
        ####################  Layer2  ###########################
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv1_ir_ = tf.pad(conv1_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv2_ir = tf.nn.conv2d(conv1_ir_, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv2_ir = lrelu(conv2_ir)
        
        with tf.variable_scope('layer2_vi'):
            weights = tf.get_variable("w2_vi", initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/w2_vi')))
            bias = tf.get_variable("b2_vi", initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/b2_vi')))
            conv1_vi_ = tf.pad(conv1_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv2_vi = tf.nn.conv2d(conv1_vi_, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv2_vi = lrelu(conv2_vi)
            print("conv2_vi:", conv2_vi.shape)
        
        ####################  Layer3  ###########################
        conv_12_ir = tf.concat([conv1_ir, conv2_ir], axis=-1)
        conv_12_vi = tf.concat([conv1_vi, conv2_vi], axis=-1)
        # print("conv_12_vi:", conv_12_vi.shape)
        #
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias = tf.get_variable("b3", initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv_12_ir = tf.pad(conv_12_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv3_ir = tf.nn.conv2d(conv_12_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer3_vi'):
            weights = tf.get_variable("w3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/w3_vi')))
            bias = tf.get_variable("b3_vi", initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/b3_vi')))
            conv_12_vi = tf.pad(conv_12_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv3_vi = tf.nn.conv2d(conv_12_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv3_vi = lrelu(conv3_vi)
            print("conv3_vi:", conv3_vi.shape)
        
        ####################  Layer4  ###########################
        conv_123_ir = tf.concat([conv1_ir, conv2_ir, conv3_ir], axis=-1)
        conv_123_vi = tf.concat([conv1_vi, conv2_vi, conv3_vi], axis=-1)
        # print("conv_123_vi:", conv_123_vi.shape)
        
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias = tf.get_variable("b4", initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv_123_ir = tf.pad(conv_123_ir, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv4_ir = tf.nn.conv2d(conv_123_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer4_vi'):
            weights = tf.get_variable("w4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/w4_vi')))
            bias = tf.get_variable("b4_vi", initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/b4_vi')))
            conv_123_vi = tf.pad(conv_123_vi, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            conv4_vi = tf.nn.conv2d(conv_123_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv4_vi = lrelu(conv4_vi)
            print("conv4_vi:", conv4_vi.shape)
        
        conv_ir_vi = tf.concat([conv1_ir, conv1_vi, conv2_ir, conv2_vi, conv3_ir, conv3_vi, conv4_ir, conv4_vi],
                               axis=-1)
        # print("conv_ir_vi:", conv_ir_vi.shape)
        
        with tf.variable_scope('layer5'):
            weights = tf.get_variable("w5", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias = tf.get_variable("b5", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_ir = tf.nn.conv2d(conv_ir_vi, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv5_ir = lrelu(conv5_ir)
            print("conv5_ir:", conv5_ir.shape)

        with tf.variable_scope('layer6'):
            weights = tf.get_variable("w6", initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6')))
            bias = tf.get_variable("b6", initializer=tf.constant(reader.get_tensor('fusion_model/layer6/b6')))
            conv6_ir = tf.nn.conv2d(conv5_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv6_ir = lrelu(conv6_ir)
            print("conv5_ir:", conv6_ir.shape)
        
        with tf.variable_scope('layer7'):
            weights = tf.get_variable("w7",  initializer=tf.constant(reader.get_tensor('fusion_model/layer7/w7')))
            bias = tf.get_variable("b7",  initializer=tf.constant(reader.get_tensor('fusion_model/layer7/b7')))
            conv7_ir = tf.nn.conv2d(conv6_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            map = tf.nn.tanh(conv7_ir) / 2 + 0.5
            print("map:", map.shape)
            mask = binary(map - 0.5)
            mask = mask / 2 + 0.5
        
    return map, mask

def binary(input):
    x = input
    # with tf.get_default_graph().gradient_override_map({"Sign": 'QuantizeGrad'}):
    x = tf.sign(x)
    return x

def input_setup(index):
    padding=0
    sub_ir_sequence = []
    sub_irtu_sequence = []
    sub_vitu_sequence = []
    sub_mask_sequence = []
    sub_vi_sequence = []


    input_ir=(imread1(data_ir[index])-127.5)/127.5
    # input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread1(data_vi[index])-127.5)/127.5
    # input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)

    input_irtu=(imread2(data_irtu[index])-127.5)/127.5
    # input_irtu=np.lib.pad(input_irtu,((padding,padding),(padding,padding)),'edge')
    w,h,c=input_irtu.shape
    input_irtu=input_irtu.reshape([w,h,c])
    sub_irtu_sequence.append(input_irtu)
    train_data_irtu= np.asarray(sub_irtu_sequence)

    input_vitu=(imread2(data_vitu[index])-127.5)/127.5
    # input_vitu=np.lib.pad(input_vitu,((padding,padding),(padding,padding)),'edge')
    w,h,c=input_vitu.shape
    input_vitu=input_vitu.reshape([w,h,c])
    sub_vitu_sequence.append(input_vitu)
    train_data_vitu= np.asarray(sub_vitu_sequence)

    input_mask=(imread1(data_mask[index])-127.5)/127.5
    # input_mask=np.lib.pad(input_mask,((padding,padding),(padding,padding)),'edge')
    w,h=input_mask.shape
    input_mask=input_mask.reshape([w,h,1])
    sub_mask_sequence.append(input_mask)
    train_data_mask= np.asarray(sub_mask_sequence)

    return train_data_ir,train_data_vi,train_data_irtu,train_data_vitu,train_data_mask

for idx_num in range(4000):
  num_epoch=idx_num
  while(num_epoch==568):

      reader = tf.train.NewCheckpointReader('./checkpoint/CGAN_84/CGAN.model-'+ str(num_epoch))

      with tf.name_scope('IR_input'):
          images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
      with tf.name_scope('VI_input'):
          images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
      # with tf.name_scope('VI_inputtu'):
      #     images_vitu = tf.placeholder(tf.float32, [1,None,None,None], name='images_vitu')
      # with tf.name_scope('IR_inputtu'):
      #     images_irtu = tf.placeholder(tf.float32, [1,None,None,None], name='images_irtu')
      with tf.name_scope('Mask_input'):
          images_mask = tf.placeholder(tf.float32, [1,None,None,None], name='images_mask')
      with tf.name_scope('input'):
          input_image_ir =images_ir
          input_image_vi =images_vi
          # input_image_irtu = images_irtu
          # input_image_vitu = images_vitu
          input_image_mask = images_mask


      with tf.name_scope('fusion'):
          fusion_map, fusion_mask =fusion_model(input_image_ir,input_image_vi)




      with tf.Session() as sess:
          init_op=tf.global_variables_initializer()
          sess.run(init_op)
          data_ir = prepare_data('Test_ir_3')
          data_vi = prepare_data('Test_vi_3')
          data_vitu = prepare_data('Test_vi_3')
          data_irtu = prepare_data('Test_ir_3')
          data_mask = prepare_data('mask')




          for i in range(len(data_ir)):
              start=time.time()
              train_data_ir, train_data_vi,train_data_irtu, train_data_vitu, train_data_mask =input_setup(i)
              # print("train_data_ir:",train_data_ir.shape)
              # print("train_data_vi:", train_data_vi.shape)
              result1 = sess.run(fusion_map,feed_dict={images_ir: train_data_ir, images_vi: train_data_vi, images_mask:train_data_mask})
              result2 = sess.run(fusion_mask, feed_dict={images_ir: train_data_ir,  images_vi: train_data_vi, images_mask:train_data_mask})
              fused_image = result2 * train_data_vitu + (1 - result2) * train_data_irtu


              print("result1:",result1.shape)
              print("result2:", result2.shape)


              result1=result1*127.5+127.5
              result2 = result2 * 127.5 + 127.5
              result2 = result2 * 127.5 + 127.5

              result1 = result1.squeeze()
              result2 = result2.squeeze()
              fused_image = fused_image.squeeze()



              print("result1:",result1.shape)
              print("result2:", result2.shape)
              print("fused_image:", fused_image.shape)

              image_path1 = os.path.join(os.getcwd(), 'result1','epoch'+str(num_epoch))
              image_path2 = os.path.join(os.getcwd(), 'result2', 'epoch' + str(num_epoch))
              image_path3 = os.path.join(os.getcwd(), 'result3', 'epoch' + str(num_epoch))
              if not os.path.exists(image_path1):
                  os.makedirs(image_path1)
              if not os.path.exists(image_path2):
                  os.makedirs(image_path2)
              if not os.path.exists(image_path3):
                  os.makedirs(image_path3)
              # if i<=9:
              #     image_path = os.path.join(image_path,'F9_0'+str(i)+".jpg")
              # else:
              image_path1 = os.path.join(image_path1,str(i)+".jpg")
              image_path2 = os.path.join(image_path2, str(i) + ".jpg")
              image_path3 = os.path.join(image_path3, str(i) + ".jpg")
              end=time.time()
              # print(out.shape)
              imsave(result1, image_path1)
              imsave(result2, image_path2)
              imsave(fused_image, image_path3)
              print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
      tf.reset_default_graph()
      num_epoch=num_epoch+1


