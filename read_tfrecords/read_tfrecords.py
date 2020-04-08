#coding=utf-8
import tensorflow as tf
import cv2

def read_and_decode(filename_queue):
    #根据文件名生成一个队列
    # filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

def run(filename):
    filename_queue = tf.train.string_input_producer([filename],num_epochs=10)
    img, label = read_and_decode(filename_queue)
    # image_batches, label_batches = tf.train.batch([img, label], batch_size=2, capacity=20)
    # 使用shuffle_batch可以随机打乱输入
    # img_batch, label_batch = tf.train.shuffle_batch([img, label],
    #                                                 batch_size=30, capacity=2000,
    #                                                 min_after_dequeue=1000)
    # print img_batch.shape
    init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        # 这是填充队列的指令，如果不执行程序会等在队列文件的读取处无法运行
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1):
            val, l = sess.run([img, label])
            # 我们也可以根据需要对val， l进行处理
            # l = to_categorical(l, 12)
            print val.shape
            cv2.imwrite('wu_{}_{}.jpg'.format(i,l),val)
            print(val.shape, l)
        coord.request_stop()
        coord.join(threads=threads)
        print('yes！')
if __name__ == '__main__':
    filename = '/Users/wuji/Documents/study/pycharm/mnist/dataset/images/MnistTfrecords/MnistTfrecords_train_00000-of-00002.tfrecord'
    run(filename)