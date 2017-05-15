import numpy as np
import cv2
import tensorflow as tf
import utils
import os
import math
import sys 

class documentTrainer:
    def __init__(self, model="5"):
        self.BATCH_SIZE = 100
        self.NO_OF_STEPS = 50000000
        self.CHECKPOINT_DIR = "../4Pointbg"+model
        self.DATA_DIR = "../../DataSet Generator/data_set"
        if (not os.path.isdir(self.CHECKPOINT_DIR)):
            os.mkdir(self.CHECKPOINT_DIR)
        Debug = True


        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.15
        self.sess = tf.Session(config=config)
        sess = self.sess
        self.train_gt=train_gt = np.load("../train_gt_bg"+model+".npy")
        train_image = np.load("../train_image_bg"+model+".npy")
        self.validate_gt=validate_gt = np.load("../validate_gt_bg"+model+".npy")
        validate_image = np.load("../validate_image_bg"+model+".npy")
        rand_list = np.random.randint(0, len(validate_image) - 1, 10)
        batch = validate_image[rand_list]

        print train_image.shape
        mean_train = np.mean(train_image, axis=(0,1,2))

        mean_train = np.expand_dims(mean_train, axis=0)
        mean_train = np.expand_dims(mean_train, axis=0)
        mean_train = np.expand_dims(mean_train, axis=0)
        print mean_train.shape
        self.train_image= train_image = train_image - mean_train
        self.validate_image = validate_image=validate_image - mean_train
        print np.mean(train_image, axis=(0,1,2))

      
        # In[ ]:

        def weight_variable(shape, name="temp"):
            initial = tf.truncated_normal(shape, stddev=0.1, name=name)
            return tf.Variable(initial)


        def bias_variable(shape, name="temp"):
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)


        # In[ ]:

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')



        with tf.name_scope("Input"):
            self.x = x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            



            x_ = tf.image.random_brightness(x, 50)
            x_ = tf.image.random_contrast(x_, lower=0.1, upper=1.9)
        with tf.name_scope("gt"):
            self.y_= y_ = tf.placeholder(tf.float32, shape=[None, 8])

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



        print h_pool5.get_shape()

        temp_size = h_pool5.get_shape()
        temp_size = temp_size[1] * temp_size[2] * temp_size[3]
        temp_size = int(temp_size)
        # In[ ]:

        print temp_size
        with tf.name_scope("FCLayers"):
            W_fc1 = weight_variable([int(temp_size), 500], name="W_fc1")
            b_fc1 = bias_variable([500], name="b_fc1")

            h_pool4_flat = tf.reshape(h_pool5, [-1, temp_size])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


            # In[ ]:

            # Adding dropout
            self.keep_prob = keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # In[ ]:


            W_fc2 = weight_variable([500, 500], name="W_fc2")
            b_fc2 = bias_variable([500], name="b_fc2")

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


            W_fc3 = weight_variable([500, 8], name="W_fc3")
            b_fc3 = bias_variable([8], name="b_fc3")

            self.y_conv =y_conv = tf.matmul(y_conv, W_fc3) + b_fc3



        # In[ ]:

        with tf.name_scope("loss"):
            self.cross_entropy=cross_entropy = tf.nn.l2_loss(y_conv - y_)

            self.mySum=mySum = tf.summary.scalar('Train_loss', cross_entropy)
            self.validate_loss=validate_loss = tf.summary.scalar('Validate_loss', cross_entropy)
        with tf.name_scope("Train"):
            self.train_step=train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)


        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('../train', sess.graph)

        self.saver=saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            #print ("PRINTING CHECKPOINT PATH")
            #print(ckpt.model_checkpoint_path)
            init = saver.restore(self.sess, ckpt.model_checkpoint_path)

        else:
            #print("Starting from scratch")
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def train(self):
        for i in range(self.NO_OF_STEPS):
            rand_list = np.random.randint(0, len(self.train_image) - 1, self.BATCH_SIZE)
            batch = self.train_image[rand_list]
            gt = self.train_gt[rand_list]
            if i % 1000== 0:
                loss_mine = self.cross_entropy.eval(feed_dict={
                    self.x: self.train_image[0:self.BATCH_SIZE], self.y_: self.train_gt[0:self.BATCH_SIZE], self.keep_prob: 1.0}, session=self.sess)
                print("Loss on Train : ", math.sqrt((loss_mine/self.BATCH_SIZE)*2))
                # summary = self.mySum.eval(feed_dict={
                #     self.x: self.train_image[0:self.BATCH_SIZE], self.y_: self.train_gt[0:self.BATCH_SIZE], self.keep_prob: 1.0})
                

                rand_list = np.random.randint(0, len(self.validate_image) - 1, self.BATCH_SIZE)
                batch = self.validate_image[rand_list]
                gt = self.validate_gt[rand_list]
                loss_mine = self.cross_entropy.eval(feed_dict={
                    self.x: batch, self.y_: gt, self.keep_prob: 1.0}, session=self.sess)
                print("Loss on Val : ", math.sqrt((loss_mine/self.BATCH_SIZE)*2))
                # val_sum = validate_loss.eval(feed_dict={
                #     x: batch, y_: gt, keep_prob: 1.0})
                
                # temp_temp = np.random.randint(0,len(validate_image)-1,1)
                # batch = validate_image[temp_temp]
                # gt = validate_gt[temp_temp]
                # response = y_conv.eval(feed_dict={
                #     x: batch, y_: gt, keep_prob: 1.0})
                # cv2.circle(batch[0], (response[0][0], response[0][1]), 2, (255,0,0),2)
                # cv2.circle(batch[0], (gt[0][0], gt[0][1]), 2, (0,255,255),2)

                # cv2.circle(batch[0], (response[0][2], response[0][3]), 2, (0,255,0),2)
                # cv2.circle(batch[0], (gt[0][2], gt[0][3]), 2, (0,255,255),2)

                # cv2.circle(batch[0], (response[0][4], response[0][5]), 2, (0,0,255),2)
                # cv2.circle(batch[0], (gt[0][4], gt[0][5]), 2, (0,255,255),2)

                # cv2.circle(batch[0], (response[0][6], response[0][7]), 2, (255,255,0),2)
                # cv2.circle(batch[0], (gt[0][6], gt[0][7]), 2, (0,255,255),2)

                # img = batch[0]
                # img = cv2.resize(img, (320,320))
                # cv2.imwrite("../temp"+str(temp_temp)+".jpg", img)
            if i % 10000== 0 and i != 0:
                self.saver.save(self.sess, self.CHECKPOINT_DIR + '/model.ckpt', global_step=i + 1)
            else:
                a= self.sess.run([self.train_step], feed_dict={self.x: batch, self.y_: gt, self.keep_prob: 1.0})


if __name__ == "__main__":
    trainer = documentTrainer(model=str(sys.argv[1]))
    trainer.train()
