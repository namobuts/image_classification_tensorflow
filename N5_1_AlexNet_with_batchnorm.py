import tensorflow as tf


def print_tensor_info(tensor):
    print('\n')
    print("tensor name:", tensor.op.name,
          "\ntensor shape:", tensor.get_shape().as_list())


def inference(input, num_class, keep_prob, is_training):
    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(input, 32, 3, (1, 1), 'same', use_bias=False, activation=tf.nn.relu)
        print_tensor_info(conv1)
        conv1_pool = tf.layers.max_pooling2d(conv1, (2, 2), 2, 'same')
        print_tensor_info(conv1_pool)
        conv1_batch = tf.layers.batch_normalization(conv1_pool, training=is_training)

    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv2d(conv1_batch, 64, 3, (1, 1), 'same', use_bias=False, activation=tf.nn.relu)
        print_tensor_info(conv2)
        conv2_pool = tf.layers.max_pooling2d(conv2, (2, 2), 2, 'same')
        print_tensor_info(conv2_pool)
        conv2_batch = tf.layers.batch_normalization(conv2_pool, training=is_training)

    with tf.variable_scope('conv3'):
        conv3 = tf.layers.conv2d(conv2_batch, 128, 3, (1, 1), 'same', use_bias=False, activation=tf.nn.relu)
        print_tensor_info(conv3)
        conv3_pool = tf.layers.max_pooling2d(conv3, (2, 2), 2, 'same')
        print_tensor_info(conv3_pool)
        conv3_batch = tf.layers.batch_normalization(conv3_pool, training=is_training)

    with tf.variable_scope('conv4'):
        conv4 = tf.layers.conv2d(conv3_batch, 256, 3, (1, 1), 'same', use_bias=False, activation=tf.nn.relu)
        print_tensor_info(conv4)
        conv4_pool = tf.layers.max_pooling2d(conv4, (2, 2), 2, 'same')
        print_tensor_info(conv4_pool)
        conv4_batch = tf.layers.batch_normalization(conv4_pool, training=is_training)

    with tf.variable_scope('conv5'):
        conv5 = tf.layers.conv2d(conv4_batch, 128, 3, (1, 1), 'same', use_bias=False, activation=tf.nn.relu)
        print_tensor_info(conv5)
        conv5_pool = tf.layers.max_pooling2d(conv5, (2, 2), 2, 'same')
        print_tensor_info(conv5_pool)
        conv5_batch = tf.layers.batch_normalization(conv5_pool, training=is_training)

    with tf.variable_scope('conv5_flatten'):
        orig_shape = conv5_batch.get_shape().as_list()
        flatten = tf.reshape(conv5_batch, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])
        print_tensor_info(flatten)

    with tf.variable_scope('fc1'):
        fc1 = tf.layers.dense(flatten, 512, use_bias=False, activation=None)
        print_tensor_info(fc1)
        fc1 = tf.layers.batch_normalization(fc1, training=is_training)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('fc2'):
        fc2 = tf.layers.dense(fc1, 128, use_bias=False, activation=None)
        print_tensor_info(fc2)
        fc2 = tf.layers.batch_normalization(fc2, training=is_training)
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob)

    with tf.variable_scope('fc3'):
        fc3 = tf.layers.dense(fc2, num_class, use_bias=False, activation=None)
        print_tensor_info(fc3)
        fc3 = tf.layers.batch_normalization(fc3, training=is_training)
        fc3 = tf.nn.relu(fc3)

    return fc3
