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

                # vgg16中只有conv和fc被计入层数，max_pool不计入

                # vgg16的第一部分。这一部分有2层conv，1层max_pool。
                net = slim.repeat(input, 2, slim.conv2d, num_outputs=64, scope='conv1')
                net = slim.max_pool2d(net, scope='pool1')
                print_tensor_info(net)

                # vgg16的第二部分。这一部分有2层conv，1层max_pool。
                net = slim.repeat(net, 2, slim.conv2d, num_outputs=128, scope='conv2')
                net = slim.max_pool2d(net, scope='pool2')
                print_tensor_info(net)

                # vgg16的第三部分。这一部分有3层conv，1层max_pool。
                net = slim.repeat(net, 3, slim.conv2d, num_outputs=256, scope='conv3')
                net = slim.max_pool2d(net, scope='pool3')
                print_tensor_info(net)

                # vgg16的第四部分。这一部分有3层conv，1层max_pool。
                net = slim.repeat(net, 3, slim.conv2d, num_outputs=512, scope='conv4')
                net = slim.max_pool2d(net, scope='pool4')
                print_tensor_info(net)

                # vgg16的第五部分。这一部分有3层conv，1层max_pool。
                net = slim.repeat(net, 3, slim.conv2d, num_outputs=512, scope='conv5')
                net = slim.max_pool2d(net, scope='pool5')
                print_tensor_info(net)

                flatten = slim.flatten(net)

                # vgg16的第六部分。这一部分有3层fc，并且前两层全连接后面还有dropout。
                net = slim.fully_connected(flatten, 1024, scope='fc1')
                # print_tensor_info(net)
                net = slim.dropout(net, keep_prob, scope='fc1_dropout')
                print_tensor_info(net)

                net = slim.fully_connected(net, 512, scope='fc2')
                # print_tensor_info(net)
                net = slim.dropout(net, keep_prob, scope='fc2_dropout')
                print_tensor_info(net)

                net = slim.fully_connected(net, num_class, scope='fc3')
                print_tensor_info(net)

    return net