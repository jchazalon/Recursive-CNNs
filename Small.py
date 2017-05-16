import numpy as np
import cv2
import tensorflow as tf

class get_corners_moreBG:
    def __init__(self, model="5"):
        CHECKPOINT_DIR = "../4Pointbg"+model

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.05
        self.sess = tf.Session(config=config)
        train_image = np.load("../train_image_bg"+model+".npy")
        mean_train = np.mean(train_image, axis=(0,1,2))

        mean_train = np.expand_dims(mean_train, axis=0)
        mean_train = np.expand_dims(mean_train, axis=0)
        self.mean_train = np.expand_dims(mean_train, axis=0)

        def weight_variable(shape, name="temp"):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)


        def bias_variable(shape, name="temp"):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)


        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')


        with tf.name_scope("Input"):
            self.inputT = self.x = x = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.x = x = tf.image.resize_images(self.x, [32, 32])
            x_ = tf.image.random_brightness(x, 5)
            x_ = tf.image.random_contrast(x_, lower=0.9, upper=1.1)
        with tf.name_scope("Conv1"):
            W_conv1 = weight_variable([5, 5, 3, 20], name="W_conv1")
            b_conv1 = bias_variable([20], name="b_conv1")
            h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
        with tf.name_scope("MaxPool1"):
            h_pool1 = max_pool_2x2(h_conv1)
        with tf.name_scope("Conv2"):
            W_conv2 = weight_variable([5, 5, 20, 40], name="W_conv2")
            b_conv2 = bias_variable([40], name="b_conv2")
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        with tf.name_scope("Conv2_1"):
            W_conv2_1 = weight_variable([5, 5, 40, 40], name="W_conv2_1")
            b_conv2_1= bias_variable([40], name="b_conv2_1")
            h_conv2_1 = tf.nn.relu(conv2d(h_conv2, W_conv2_1) + b_conv2_1)
        with tf.name_scope("MaxPool2"):
            h_pool2 = max_pool_2x2(h_conv2_1)
        with tf.name_scope("Conv3"):
            W_conv3 = weight_variable([5, 5, 40, 60], name="W_conv3")
            b_conv3 = bias_variable([60], name="b_conv3")
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

            W_conv3_1 = weight_variable([5, 5, 60, 60], name="W_conv3_1")
            b_conv3_1 = bias_variable([60], name="b_conv3_1")
            h_conv3_1 = tf.nn.relu(conv2d(h_conv3, W_conv3_1) + b_conv3_1)
        with tf.name_scope("MaxPool3"):
            h_pool3 = max_pool_2x2(h_conv3_1)
        with tf.name_scope("Conv4"):
            W_conv4 = weight_variable([5, 5, 60, 80], name="W_conv4")
            b_conv4 = bias_variable([80], name="b_conv4")
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        with tf.name_scope("Maxpool4"):
            h_pool4 = max_pool_2x2(h_conv4)
        with tf.name_scope("Conv5"):
            W_conv5 = weight_variable([5, 5, 80, 100], name="W_conv5")
            b_conv5 = bias_variable([100], name="b_conv5")
            h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5)


        temp_size = h_pool5.get_shape()
        temp_size = temp_size[1] * temp_size[2] * temp_size[3]
        temp_size = int(temp_size)

        with tf.name_scope("FCLayers"):
            W_fc1 = weight_variable([int(temp_size), 500], name="W_fc1")
            b_fc1 = bias_variable([500], name="b_fc1")

            h_pool4_flat = tf.reshape(h_pool5, [-1, temp_size])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


            self.keep_prob = keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


            W_fc2 = weight_variable([500, 500], name="W_fc2")
            b_fc2 = bias_variable([500], name="b_fc2")

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


            W_fc3 = weight_variable([500, 8], name="W_fc3")
            b_fc3 = bias_variable([8], name="b_fc3")

            self.y_conv =y_conv = tf.matmul(y_conv, W_fc3) + b_fc3



        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            init = saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
    def get(self,img):
        img_temp = np.expand_dims(img, axis=0)
        img_temp = img_temp - self.mean_train
        # print self.mean_train
        # [[[[ 146.55914183  139.54338589  154.33292928]]]]

        response = self.y_conv.eval(feed_dict={
            self.inputT: img_temp, self.keep_prob: 1.0}, session=self.sess)
        response = response[0]/32
        x = response[[0,2,4,6]]
        y = response[[1,3,5,7]]
        x = x*img.shape[1]
        y = y*img.shape[0]
        return x, y

if __name__ == "__main__":
	cornerExt = get_corners_moreBG("6")
	img = cv2.imread("001.jpg")
        print cornerExt.get(img)

