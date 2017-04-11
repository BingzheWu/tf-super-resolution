from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge
)

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
slim =  tf.contrib.slim

class SRCNN(object):
  def __init__(self, 
               sess, 
               image_size=33,
               label_size=17, 
               batch_size=128,
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
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    self.pred = self.model()
    # Loss function (MSE)
    self.tv =tf.image.total_variation(self.pred)
    print(type(self.pred))
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))+0.0005*tf.reduce_mean(self.tv)
    self.saver = tf.train.Saver()

  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config)
    else:
      nx, ny = input_setup(self.sess, config)

    if config.is_train:     
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

    train_data, train_label = read_data(data_dir)

    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 500 == 0:
            self.save(config.checkpoint_dir, counter)

    else:
      print("Testing...")

      result = self.pred.eval({self.images: train_data, self.labels: train_label})

      result = merge(result, [nx, ny])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      print(train_label.shape)
      self.imshow(result)
      imsave(result, image_path)
  def input_setup(self, label, config):
      import scipy.ndimage
      input_ = scipy.ndimage.interpolation.zoom(label, 1.0/3.0)
      input_ = scipy.ndimage.interpolation.zoom(input_, 3.)
      h, w = input_.shape
      print(h,w)
      sub_input_sequence = []
      sub_label_sequence = []
      nx = ny = 0
      for x in range(0, h-config.image_size+1, config.stride):
          nx+=1
          ny=0
          for y in range(0, w-config.image_size+1, config.stride):
              ny+=1
              sub_input = input_[x:x+config.image_size, y:y+config.image_size]
              sub_label = label[x:x+config.label_size, y:y+config.label_size]
              sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
              sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
              sub_input_sequence.append(sub_input)
              sub_label_sequence.append(sub_label)
      arrdata = np.asarray(sub_input_sequence)
      arrlabel = np.asarray(sub_label_sequence)
      return arrdata, arrlabel, nx, ny
  def test(self, img_path, config):
      import scipy.misc
      ycbcr = scipy.misc.imread(img_path, mode = 'YCbCr').astype(float)
      label = ycbcr[:,:,0]
      input_ = scipy.ndimage.interpolation.zoom(label, 1.0/3.0)
      input_ = scipy.ndimage.interpolation.zoom(input_, 3.)
      batch_data, batch_label, nx, ny= self.input_setup(label, config)
      self.sess.run(tf.initialize_all_variables())
      self.load(self.checkpoint_dir)
      result = self.sess.run(self.pred, feed_dict = {self.images: batch_data, self.labels: batch_label})
      result = merge(result, [nx,ny])
      result = result.squeeze()
      print(result.shape)
      self.imshow(input_)
      self.imshow(result)
  def imshow(self, img):
      import matplotlib.pyplot as plt
      plt.imshow(img, cmap = 'gray')
      plt.show()
  def model(self):
    with slim.arg_scope([slim.conv2d], padding = 'VALID', weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=slim.l2_regularizer(0.00005)):
        conv1 = slim.conv2d(self.images, 64, [9,9], scope = 'conv1')
        conv2 = slim.conv2d(conv1, 32, [5,5], scope = 'conv2')
        conv3 = slim.conv2d(conv2, 1, [5,5], scope = 'conv3', activation_fn = None, normalizer_fn = None)
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

class DCGAN(object):
    def __init__(self, sess, image_size = 128, is__crop = True,
                batch_size = 64,image_shape = [128, 128, 3], y_dim = None,
                z_dim = 100, gf_dim = 64, df_dim = 64,
                gfc_dim = 1024, dfc_dim = 1024, c_dim = 3, dataset_name = 'default',
                 checkpoint_dir = None):
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_size = 32
        self.sample_size = image_shape
        self.y_dim  = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3],
                                     name = 'read_images')
        self.up_inputs = tf.image.resize_images(self.inputs, [self.image_shape[0], self.image_shape[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.images = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape, name = 'real_images')
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size]+self.image_shape,
                                           name= 'sample_images')
        self.G = self.generator(self.inputs)
        self.G_sum = tf.image_summary("G", self.G)

        self.g_loss = tf.reduce_mean(tf.square(self.images-self.G))

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)

        t_vars = tf.trainble_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        data = sorted(glob(os.path.join("./data", config.dataset, "valid", "*.jpg")))

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1).minimize(self.g_loss, var_list = self.g_vars)
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        self.g_sum = tf.merge_summary([self.G_sum, self.g_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop = self.is_crop) for sample_file in sample_files]
        sample_inputs = [doresize(xx, [self.input_size]*2) for xx in sample]
        sample_images = np.array(sample).astype(np.float32)
        sample_input_image = np.array(sample_inputs).astype(np.float32)

        counter = 1
        start_time = time.time()

        for epoch in xrange(config.epoch):
            data = sorted(glob(os.path.join("./data", config.dataset, "train", "*.jpg")))
            batch_idxs = min(len(data), config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop = self.is_crop) for batch_file in batch_files]
                input_batch = [doresize(xx, [self.input_size]*2) for xx in batch]
                batch_images = np.array(batch).astype(np.float32)
                batch_inputs = np.array(input_batch).astype(np.float32)

                _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss], 
                                                     feed_dict = {self.inputs: batch_inputs, self.images: batch_images})
                self.writer.add_summary(summary_str, counter)
                counter+=1
                if np.mod(counter, 500)==2:
                    self.save(config.checkpoint_dir, counter)

    def generator(self, z):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_initializer = tf.contrib.layers.xavier_initializer_conv2d()):
            h0 = slim.conv2d_transpose(z, num_outputs = self.gf_dim, activation_fn = tf.nn.relu, normalizer_fn = None)
            h1 = slim.conv2d_transpose(h0, num_outputs = self.gf_dim, activation_fn = tf.nn.relu, normalizer_fn = None)
            h2 = slim.conv2d_transpose(h1, num_outputs = 3*16)
            return tf.nn.tanh(h2)
