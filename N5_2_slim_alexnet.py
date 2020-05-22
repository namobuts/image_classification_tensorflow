import tensorflow as tf
import tensorflow.contrib.slim as slim


def print_tensor_info(tensor):
    print('\n')
    print("tensor name:", tensor.op.name,
          "\ntensor shape:", tensor.get_shape().as_list())


def inference(input, num_class, keep_prob, is_training):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training}):
        with slim.arg_scope([slim.conv2d], stride=1, kernel_size=[3, 3], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], stride=2, kernel_size=[2, 2], padding='SAME'):

                # alexnet的第1层
                conv1 = slim.conv2d(input, num_outputs=32, scope='conv1')
                pool1 = slim.max_pool2d(conv1, scope='pool1')
                print_tensor_info(pool1)

                # alexnet的第2层
                conv2 = slim.conv2d(pool1, num_outputs=64, scope='conv2')
                pool2 = slim.max_pool2d(conv2, scope='pool2')
                print_tensor_info(pool2)

                # alexnet的第3层
                conv3 = slim.conv2d(pool2, num_outputs=128, scope='conv3')
                pool3 = slim.max_pool2d(conv3, scope='pool3')
                print_tensor_info(pool3)

                # alexnet的第4层
                conv4 = slim.conv2d(pool3, num_outputs=256, scope='conv4')
                pool4 = slim.max_pool2d(conv4, scope='pool4')
                print_tensor_info(pool4)

                # alexnet的第5层
                conv5 = slim.conv2d(pool4, num_outputs=128, scope='conv5')
                pool5 = slim.max_pool2d(conv5, scope='pool5')
                print_tensor_info(pool5)

                flatten = slim.flatten(pool5)

                # alexnet的第6层
                fc1 = slim.fully_connected(flatten, 1024, scope='fc1')
                print_tensor_info(fc1)
                fc1 = slim.dropout(fc1, keep_prob, scope='fc1_dropout')

                # alexnet的第7层
                fc2 = slim.fully_connected(fc1, 512, scope='fc2')
                print_tensor_info(fc2)
                fc2 = slim.dropout(fc2, keep_prob, scope='fc2_dropout')

                # alexnet的第8层
                fc3 = slim.fully_connected(fc2, num_class, scope='fc3')
                print_tensor_info(fc3)

    return fc3

