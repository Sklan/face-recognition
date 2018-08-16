from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import os.path
import re
import tarfile

import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    result = collections.OrderedDict()
    sub_dirs = [os.path.join(image_dir, item) for item in gfile.ListDirectory(image_dir)]
    sub_dirs = sorted(item for item in sub_dirs if gfile.IsDirectory(item))

    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {'dir': dir_name, 'training': training_images, 'testing': testing_images,
                              'validation': validation_images}
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '.txt'


def create_model_graph(model_info):
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(model_dir, model_info['model_file_name'])
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[
                model_info['bottleneck_tensor_name'], model_info['resized_input_tensor_name']]))
    return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def download_and_extract(data_url):
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(data_url, filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                                                resized_input_tensor, bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir,
                             jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category)

    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess,
                               jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess,
                               jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor):
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir,
                                         jpeg_data_tensor, decoded_image_tensor, resized_input_tensor,
                                         bottleneck_tensor)
            how_many_bottlenecks += 1


def get_random_cached_bottlenecks(sess, image_lists, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if category == 'training':
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
                if len(filenames) == 32:
                    break
    else:
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)

    return bottlenecks, ground_truths, filenames


def add_final_training_ops(learning_rate, class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[None, bottleneck_tensor_size],
                                                       name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor


def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [final_tensor_name])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def create_model_info():
    return {'data_url': 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz',
            'bottleneck_tensor_name': 'pool_3/_reshape:0',
            'bottleneck_tensor_size': 2048,
            'input_width': 299,
            'input_height': 299,
            'input_depth': 3,
            'resized_input_tensor_name': 'Mul:0',
            'model_file_name': 'classify_image_graph_def.pb',
            'input_mean': 128,
            'input_std': 128}


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


image_dir = 'data/images'
model_dir = 'data/imagenet'
output_graph = 'data/output_graph_.pb'
output_labels = 'data/output_labels.txt'
bottleneck_dir = 'data/bottleneck'
final_tensor_name = 'final_result'

learning_rate = 0.3
steps = 10
testing_percentage = 20
validation_percentage = 15

model_info = create_model_info()

download_and_extract(model_info['data_url'])
graph, bottleneck_tensor, resized_image_tensor = (create_model_graph(model_info))

# Look at the folder structure, and create lists of all the images.
image_lists = create_image_lists(image_dir, testing_percentage, validation_percentage)

class_count = len(image_lists.keys())

with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(model_info['input_width'], model_info['input_height'],
                                                               model_info['input_depth'], model_info['input_mean'],
                                                               model_info['input_std'])

    cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                      resized_image_tensor, bottleneck_tensor)

    # Add the new layer for training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(learning_rate,
                                            len(image_lists.keys()), final_tensor_name, bottleneck_tensor,
                                            model_info['bottleneck_tensor_size'])

    evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the training
    for step in range(steps):
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
            sess, image_lists, 'training',
            bottleneck_dir, image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor)
        # Feed the bottlenecks and ground truth into the graph, and run a training step.
        _ = sess.run(
            [train_step],
            feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, 'validation',
                bottleneck_dir, image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor))
        # Run a validation step
        validation_accuracy = sess.run([evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks,
                                                                     ground_truth_input: validation_ground_truth})
        print(step, validation_accuracy)

    test_bottlenecks, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(
        sess, image_lists, 'testing', bottleneck_dir, image_dir, jpeg_data_tensor, decoded_image_tensor,
        resized_image_tensor, bottleneck_tensor))
    test_accuracy, predictions = sess.run([evaluation_step, prediction],
                                          feed_dict={bottleneck_input: test_bottlenecks,
                                                     ground_truth_input: test_ground_truth})
    print(test_accuracy)
    save_graph_to_file(sess, graph, output_graph)
    with gfile.FastGFile(output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')
