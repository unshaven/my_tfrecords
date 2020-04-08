#coding=utf-8
import math
import os
import tensorflow as tf
import sys
import numpy as np
import time
from PIL import Image
slim = tf.contrib.slim

#State the labels filename
LABELS_FILENAME = 'labels.txt'
#===================================================  Dataset Utils  ===================================================

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image_raw': bytes_feature(image_data),
      'format': bytes_feature(image_format),
      'label': int64_feature(class_id),
      'height': int64_feature(height),
      'width': int64_feature(width),
  }))

def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  suffix = 'MnistTfrecords'
  dataset_root = os.path.dirname(dataset_dir)
  tfrecords_dir = os.path.join(dataset_root,suffix)

  labels_filename = os.path.join(tfrecords_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'r') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names

#=======================================  Conversion Utils  ===================================================

#Create an image reader object for easy reading of the images
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    # channel = 1
    assert image.shape[2] == 1
    # channel = 3
    # assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  """
  flowers\
    flower_photos\
        tulips\
            ....jpg
            ....jpg
            ....jpg
        sunflowers\
            ....jpg
        roses\
            ....jpg
        dandelion\
            ....jpg
        daisy\
            ....jpg
  Note: Your dataset_dir should be /path/to/flowers and not /path/to/flowers/flowers_photos
  dataset_main_folder_list = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,name))]
  dataset_root = os.path.join(dataset_dir, dataset_main_folder_list[0])
  """
  dataset_root = dataset_dir
  directories = []
  class_names = []
  for filename in os.listdir(dataset_root):
    path = os.path.join(dataset_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)

def _get_filenames_and_classes_mnist(dataset_dir):
  dataset_root = dataset_dir
  class_names = []
  images = []
  for filename in os.listdir(dataset_root):
    label = filename[filename.find('[') + 1]
    images.append(os.path.join(dataset_root, filename))
    if label in class_names:
      continue
    class_names.append(label)
  return images, sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      tfrecord_filename, split_name, shard_id, _NUM_SHARDS)
  suffix = 'TestMnistTfrecords'
  dataset_root = os.path.dirname(dataset_dir)
  tfrecords_dir = os.path.join(dataset_root,suffix)
  if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)
  return os.path.join(tfrecords_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, tfrecord_filename, _NUM_SHARDS):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = _NUM_SHARDS)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()
            # print "filenames[i]=",filenames[i]
            img = Image.open(filenames[i],'r')
            img_raw = img.tobytes()
            # image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            #image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            # try:
            height,width = img.size[0],img.size[1]
            # height, width = image_reader.read_image_dims(sess, image_data)
            # except:
            #   sys.stdout.write("错误的图片为:{}".format(os.path.basename(filenames[i])))
            #   continue
            #class_name = os.path.basename(os.path.dirname(filenames[i]))

            class_name = os.path.basename(filenames[i])
            class_name = class_name[class_name.find('[')+1]
            class_id = class_names_to_ids[class_name]

            # example = image_to_tfexample(
            #     image_data, 'jpg', height, width, class_id)
            example = image_to_tfexample(
                img_raw, 'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir, _NUM_SHARDS, output_filename):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      tfrecord_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id, output_filename, _NUM_SHARDS)
      if not tf.gfile.Exists(tfrecord_filename):
        return False
  return True




