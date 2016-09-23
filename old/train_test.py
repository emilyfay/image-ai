import tensorflow as tf

from fill_deconv import DCGAN

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=64, batch_size=1,
                  is_crop=False, checkpoint_dir="final_checkpoint_deconv")

    filled_image = dcgan.train()




