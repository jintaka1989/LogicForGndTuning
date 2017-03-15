# -*- coding: utf-8 -*-
# import pdb; pdb.set_trace()
import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import tensorflow.python.platform
from types import *
import time
import glob
import re

# read config.ini
import ConfigParser
inifile = ConfigParser.SafeConfigParser()
inifile.read('./config.ini')
NUM_CLASSES = int(inifile.get("settings", "num_classes"))
DOWNLOAD_LIMIT = int(inifile.get("settings", "download_limit"))

IMAGE_SIZE = int(inifile.get("settings", "image_size"))
IMAGE_CHANNEL = int(inifile.get("settings", "image_channel"))
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL

MAX_STEPS = int(inifile.get("settings", "max_steps"))
FOR_OPTIMIZER = float(inifile.get("settings", "for_optimizer"))

POOL_TIMES = int(inifile.get("settings", "pool_times"))
POOL_SIZE = int(inifile.get("settings", "pool_size"))
REDUCTION = POOL_SIZE*POOL_TIMES

class NeuralNet(object):
    """docstring for NeuralNet."""
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flags = tf.app.flags
        self.FLAGS = self.flags.FLAGS
        self.flags.DEFINE_string('readmodels', 'models/model.ckpt', 'File name of model data')
        self.flags.DEFINE_string('train', 'data_set/train.txt', 'File name of train data')
        self.flags.DEFINE_string('test', 'data_set/test.txt', 'File name of test data')
        self.flags.DEFINE_string('train_dir', '/tmp/pict_data', 'Directory to put the data_set data.')
        self.flags.DEFINE_integer('max_steps', MAX_STEPS, 'Number of steps to run trainer.')
        self.flags.DEFINE_integer('batch_size', 256, 'Batch size'
                             'Must divide evenly into the dataset sizes.')
        self.flags.DEFINE_float('learning_rate', FOR_OPTIMIZER, 'Initial learning rate.')
        self.flags.DEFINE_string('classlist', 'data_set/classlist.txt', 'File name of each class data')

        self.images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        self.keep_prob = tf.placeholder("float")

        self.logits = self.inference(self.images_placeholder, self.keep_prob)
        sess = tf.InteractiveSession()

        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,self.FLAGS.readmodels)

    def preprocess(self, img):
        # # ガンマ定数の定義
        # gamma = 2.0
        # look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
        #
        # for i in range(256):
        #     look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
        #
        # # ガンマ変換後の出力
        # img = cv2.LUT(img, look_up_table)

        # RGB空間からグレースケール空間
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def inference(self,images_placeholder, keep_prob):
        # 入力をIMAGE_SIZExIMAGE_SIZExIMAGE_CHANNELに変形
        x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)

        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('conv1') as scope:
            W_conv1 = weight_variable([5, 5, IMAGE_CHANNEL, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        with tf.name_scope('pool1') as scope:
            h_pool1 = max_pool_2x2(h_conv1)

        with tf.name_scope('conv2') as scope:
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        with tf.name_scope('pool2') as scope:
            h_pool2 = max_pool_2x2(h_conv2)

        with tf.name_scope('fc1') as scope:
            W_fc1 = weight_variable([(IMAGE_SIZE/REDUCTION)*(IMAGE_SIZE/REDUCTION)*64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, (IMAGE_SIZE/REDUCTION)*(IMAGE_SIZE/REDUCTION)*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        with tf.name_scope('fc2') as scope:
            W_fc2 = weight_variable([1024, NUM_CLASSES])
            b_fc2 = bias_variable([NUM_CLASSES])

        with tf.name_scope('softmax') as scope:
            y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        return y_conv

    def classificate_one_face(self, face_image):
        font_color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_PLAIN
        font_size = 1

        with open(self.FLAGS.classlist, 'r') as f: # train.txt
            classlist = []
            for line in f:
                line = line.rstrip()
                l = line.split()
                classlist.append(l[1])
        face_image = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE))
        face_image = face_image.flatten().astype(np.float32)/255.0
        face_image = np.asarray(face_image)
        face_text = ""
        pr = self.logits.eval(feed_dict={
            self.images_placeholder: [face_image],
            self.keep_prob: 1.0 })[0]
        pred = np.argmax(pr)
        # print pr
        _max=max(pr)
        # print str(pred) + "({:.1%})".format(_max)
        if _max >=0.80:
            face_text = (classlist[pred] + "({:.0%})".format(_max))
            # face_text = (classlist[pred] + "({:.1%})".format(_max))
        elif _max<0.80:
            face_text = ("?")
        # cv2.putText(frame,face_text,(10,10*(i*2+1)),font, font_size,font_color)
        # cv2.putText(face_image,face_text,(5, 5),font, font_size,font_color)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return face_text

    def create_cascade(self):
        # サンプル顔認識特徴量ファイル
        cascade_path = "haarcascades/haarcascade_frontalface_alt.xml"
            # cascade_path = "/host/Users/YA65857/Downloads/opencv/sources/data/hogcascades/hogcascade_pedestrians.xml"
        # 分類器を作る作業
        cascade = cv2.CascadeClassifier(cascade_path)
        return cascade

    def classificate_face(self,mirror=True, size=None):
        # これは、BGRの順になっている気がする
        color = (255, 255, 255) #白
        font_color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_PLAIN
        font_size = 1

        cascade = self.create_cascade()
        # 保存先
        dir_path = "class"
        files = glob.glob(dir_path + "/*")
        if len(files) == 0:
            i = 0
        else:
            i = int(max(files).replace(".jpg", "").replace("tmp/", ""))

        with open(self.FLAGS.classlist, 'r') as f: # train.txt
            classlist = []
            for line in f:
                line = line.rstrip()
                l = line.split()
                classlist.append(l[1])

        # カメラをキャプチャする
        cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号

        # 1回目の画像取得
        ret, frame = cap.read()

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        cv2.imshow("image", frame)

        while True:
            # retは画像を取得成功フラグ
            ret, frame = cap.read()

            # グレースケール変換
            gray = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)

            # 顔認識の実行
            facerect = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1, minSize=(10, 10), maxSize=(480,480))

            face_images = []

            if len(facerect) > 0:
                # 検出した顔を囲む矩形の作成
                # 検出した人物の名前をつける
                for rect in facerect:
                    face_image = self.cut_face(frame, rect)
                    # 前処理
                    face_image = self.preprocess(face_image)
                    if IMAGE_CHANNEL == 1:
                        # グレースケール変換
                        if cv2.__version__ == '3.0.0':
                            face_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # opencv3.0
                        else:
                            face_image = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY) # opencv2.4...
                    face_image = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE))
                    face_images.append(face_image.flatten().astype(np.float32)/255.0)
                    cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                    # cv2.putText(frame,"recognizing face",(10,10),font, font_size,font_color)

                face_images = np.asarray(face_images)
                face_text = ""
                for i in range(len(face_images)):
                    rect = facerect[i]
                    pr = self.logits.eval(feed_dict={
                        self.images_placeholder: [face_images[i]],
                        self.keep_prob: 1.0 })[0]
                    pred = np.argmax(pr)
                    # print pr
                    _max=max(pr)
                    # print str(pred) + "({:.1%})".format(_max)
                    if _max >=0.80:
                        face_text = (classlist[pred] + "({:.0%})".format(_max))
                        # face_text = (classlist[pred] + "({:.1%})".format(_max))
                    elif _max<0.80:
                        face_text = ("?")

                    # cv2.putText(frame,face_text,(10,10*(i*2+1)),font, font_size,font_color)
                    cv2.putText(frame,face_text,tuple(rect[0:2]-5),font, font_size,font_color)
            else:
                print("no face")

            # フレームを表示する(時間も計測する)
            cv2.imshow("image", frame)

            k = cv2.waitKey(1) # 1msec待つ
            if k == 27: # ESCキーで終了
                break
                # elif i == 800: # この枚数保存で終了
                # break

        # キャプチャを解放する
        cap.release()
        cv2.destroyAllWindows()

    def cut_and_save(self,image, path, rect):
        # 顔だけ切り出して保存
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        dst = image[y:y + height, x:x + width]
        #認識結果の保存
        cv2.imwrite(path, dst)

    def cut_face(self, image, rect):
        # 顔だけ切り出して返す
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        dst = image[y:y + height, x:x + width]
        return dst

if __name__ == "__main__":
    img = cv2.imread("tmp/147.jpg", 1)
    net = NeuralNet()
    # net.classificate_face()
    print net.classificate_one_face(img)

    # net2 = NeuralNet()
    # # net.classificate_face()
    # print net2.classificate_one_face(img)
