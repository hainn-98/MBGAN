import tensorflow as tf
from Datasets.DataHandler import  DataHandler
from Model.MBGAN import MBGAN
import numpy as np


if __name__ == '__main__':
	seed = 999
	np.random.seed(seed)
	tf.set_random_seed(seed)
	tf.compat.v1.disable_eager_execution()
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	# log('Start')
	handler = DataHandler()
	handler.prepare_data()
	# log('Load Data')

	with tf.compat.v1.session(config=config) as sess:
		recom = MBGAN(sess, handler)
		recom.run()