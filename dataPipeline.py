import tensorflow as tf
import glob
import os
import numpy as np
import time
# tf.enable_eager_execution()

batch_size = 10
directory =  '/hdd_2/cifar-100-python/data/train/fish/'
target_size = (64,64)


def get_files(dir_path, label):
    globbed = tf.string_join([dir_path, '*.png'])
    files = tf.matching_files(globbed)
    num_files = tf.shape(files)[0]  # in the directory
    labels = tf.tile([label], [num_files, ])  # expand label to all files

    return tf.data.Dataset.from_tensor_slices((files,labels))

def read_decode(path,label):
    img = tf.image.decode_image(tf.read_file(path), channels=3)
    img = tf.image.resize_bilinear(tf.expand_dims(img, axis=0), target_size)
    img = tf.squeeze(img, 0)
    label = tf.one_hot(label, num_classes)
    img = tf.Print(img, [path, label], 'Read_decode')
    return img, label

classes = sorted(glob.glob(directory + '/*/')) # final slash selects directories only
num_classes = len(classes)
labels = np.arange(0,num_classes)
dirs = tf.data.Dataset.from_tensor_slices((classes,labels))

files = dirs.apply(tf.contrib.data.parallel_interleave(get_files, cycle_length=5, sloppy=False))
imgs = files.apply(tf.contrib.data.shuffle_and_repeat(1000))
imgs = imgs.apply(tf.contrib.data.map_and_batch(read_decode, batch_size, num_parallel_batches=4))
# imgs = imgs.map(read_decode,num_parallel_calls=4).batch(batch_size)
imgs = imgs.prefetch(4)

imgs = imgs.make_one_shot_iterator().get_next()


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    while True:
        try:
            tt = time.time()
            for i in range(0,100):
                t,a = sess.run(imgs)
            print(time.time()-tt)
        except tf.errors.OutOfRangeError:
            break

# #data pipeline ex 1
#
# def img_decode(img):
#     img = img - 128
#     return img
#
# x = np.random.sample((100, 2))
# dataset = tf.data.Dataset.from_tensor_slices(x)
# dataset = dataset.shuffle(10000)
# dataset = dataset.repeat()
# dataset = dataset.map(img_decode, num_parallel_calls=3)
# dataset = dataset.batch(5)\
#     .make_one_shot_iterator()
#
# imgs = dataset.get_next()

# with tf.Session() as sess:
#     while True:
#         try:
#             img, label = sess.run(imgs)
#         except tf.errors.OutOfRangeError:
#             break