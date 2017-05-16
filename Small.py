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

        print self.mean_train
        # [[[[ 146.55914183  139.54338589  154.33292928]]]]
        # (array([  698.175354  ,  1161.4095459 ,  1302.14160156,   768.1395874 ], dtype=float32), array([ 181.6534729 ,  208.27745056,  821.59490967,  859.08441162], dtype=float32))
        # [[[[ 146.55914183  139.54338589  154.33292928]]]]
        # (1080, 1920, 3)
        # (array([  697.51312256,  1160.88891602,  1302.05566406,   768.35913086], dtype=float32), array([ 181.32633972,  208.20285034,  821.5411377 ,  859.25219727], dtype=float32))
        # (array([  700.22625732,  1163.03771973,  1303.61547852,   767.75402832], dtype=float32), array([ 183.30375671,  207.90759277,  821.47290039,  858.97912598], dtype=float32))
        # (array([  700.51947021,  1131.64453125,  1302.01855469,   763.37438965], dtype=float32), array([ 192.44558716,  203.4969635 ,  835.41467285,  871.97808838], dtype=float32))
        # (array([  700.51947021,  1131.64453125,  1302.01855469,   763.37438965], dtype=float32), array([ 192.44558716,  203.4969635 ,  835.41467285,  871.97808838], dtype=float32))
        # (array([  700.22625732,  1163.03771973,  1303.61547852,   767.75402832], dtype=float32), array([ 183.30375671,  207.90759277,  821.47290039,  858.97912598], dtype=float32))
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
            self.inputT = self.x = x = tf.placeholder(tf.float32, shape=[None, None, 3])
            mean = tf.constant([[[146.55914183,  139.54338589,  154.33292928]]])
            self.x = x = tf.image.resize_images(self.x, [32, 32])
            self.x = x = tf.subtract(self.x, mean)
            self.x = x  =tf.reshape(self.x, [1,32,32,3])
        with tf.name_scope("Conv1"):
            W_conv1 = weight_variable([5, 5, 3, 20], name="W_conv1")
            b_conv1 = bias_variable([20], name="b_conv1")
            h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
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

        with tf.name_scope("FCLayers"):
            W_fc1 = weight_variable([100, 500], name="W_fc1")
            b_fc1 = bias_variable([500], name="b_fc1")

            h_pool4_flat = tf.reshape(h_pool5, [-1, 100])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


            h_fc1_drop = h_fc1


            W_fc2 = weight_variable([500, 500], name="W_fc2")
            b_fc2 = bias_variable([500], name="b_fc2")

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


            W_fc3 = weight_variable([500, 8], name="W_fc3")
            b_fc3 = bias_variable([8], name="b_fc3")

            self.y_conv =y_conv = tf.matmul(y_conv, W_fc3) + b_fc3
            div_cont = tf.constant([32.0])
            self.scaledoutput = tf.divide(self.y_conv, div_cont)



        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            init = saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        tf.train.write_graph(self.sess.graph_def, "../", "mymodel.pbtxt")
    def get(self,img_temp):
        # img_temp = img_temp - self.mean_train[0]

        response = self.scaledoutput.eval(feed_dict={
            self.inputT: img_temp}, session=self.sess)
        # response = response[0]/32
        print "Reponse", response
        response = response[0]
        x = response[[0,2,4,6]]
        y = response[[1,3,5,7]]
        x = x*img.shape[1]
        y = y*img.shape[0]
        return x, y

if __name__ == "__main__":
	cornerExt = get_corners_moreBG("6")
	img = cv2.imread("001.jpg")
        print cornerExt.get(img)

