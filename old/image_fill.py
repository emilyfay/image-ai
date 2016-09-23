#!/usr/bin/env python2.7

import numpy as np
import tensorflow as tf
from scipy import ndimage

from Image_completer.utils import imsave_single
from fill_deconv import DCGAN
print('next')
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_string("dataset", "./data/masked", "Dataset directory.")
flags.DEFINE_string("filename", "test.png", "file to load")
flags.DEFINE_string("checkpoint_dir", "final_checkpoint_deconv", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "filled", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

image_path = "static/masked_images/"+FLAGS.filename
masked_image = ndimage.imread(image_path, mode='RGB')
print('got image')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)

    filled_image = dcgan.infill(masked_image)

imsave_single(filled_image, ("static/completed_images/out_%s" %filename))
print('saved completed image as %s' %filename)



