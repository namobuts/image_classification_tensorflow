import tensorflow as tf
import tensorflow.contrib.slim as slim


def print_tensor_info(tensor):
    print('\n')
    print("tensor name:", tensor.op.name,
          "\ntensor shape:", tensor.get_shape().as_list())


def inception_module(previous_layer):

    net = previous_layer

    # 给inception_module一个统一的变量空间
    with tf.variable_scope('inception_module'):

        # 给module的每一个部分定义一个变量空间

        # 第一部分。这部分只有1x1卷积
        with tf.variable_scope('branch_0'):

            branch_0 = slim.conv2d(net, 128, kernel_size=[1, 1], padding='SAME', stride=1,
                                   activation_fn=tf.nn.relu, scope='conv2d_1x1')

        # 第二部分。这部分先是1x1卷积，然后再接3x3卷积。
        with tf.variable_scope('branch_1'):

            branch_1_0 = slim.conv2d(net, 64, kernel_size=[1, 1], padding='SAME', stride=1,
                                     activation_fn=tf.nn.relu, scope='conv2d_1x1')

            branch_1_1 = slim.conv2d(branch_1_0, 96, kernel_size=[3, 3], padding='SAME', stride=1,
                                     activation_fn=tf.nn.relu, scope='conv2d_3x3')

        # 第三部分。这部分先是1x1卷积，然后再接5x5卷积。
        with tf.variable_scope('branch_2'):

            branch_2_0 = slim.conv2d(net, 16, kernel_size=[1, 1], padding='SAME', stride=1,
                                     activation_fn=tf.nn.relu, scope='conv2d_1x1')

            branch_2_1 = slim.conv2d(branch_2_0, 32, kernel_size=[5, 5], padding='SAME', stride=1,
                                     activation_fn=tf.nn.relu, scope='conv2d_5x5')

        # 第四部分。这部分先是3x3的max_pool，然后再接1x1卷积。
        with tf.variable_scope('branch_3'):

            branch_3_0 = slim.max_pool2d(net, kernel_size=[3, 3], padding='SAME', stride=1,
                                         scope='max_pool2d_3x3')

            branch_3_1 = slim.conv2d(branch_3_0, 32, kernel_size=[1, 1], padding='SAME', stride=1,
                                     activation_fn=tf.nn.relu, scope='conv2d_1x1')

        # 将这几个部分concatenation，得到最终的inception_module输出
        with tf.variable_scope('concatenation'):
            # tf.concatd 最后一个数字代表在合并维度即[n,h,w,c]（[0,1,2,3]）。3代表在channel维度上合并。
            net = tf.concat([branch_0, branch_1_1, branch_2_1, branch_3_1], 3)
            print_tensor_info(net)

    return net


def inference(input, num_class, keep_prob, is_training):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training}):
        with slim.arg_scope([slim.conv2d], stride=1, kernel_size=[3, 3], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], stride=2, kernel_size=[2, 2], padding='SAME'):

                # alexnet的第1层
                conv1 = slim.conv2d(input, num_outputs=128, scope='conv1')
                pool1 = slim.max_pool2d(conv1, scope='pool1')
                print_tensor_info(pool1)

                # 将原先alexnet的2，3，4层卷积层用inception_module替换

                '''
                # alexnet的第2层
                conv2 = slim.conv2d(pool1, num_outputs=64, scope='conv2')
                pool2 = slim.max_pool2d(conv2, scope='pool2')
                print_tensor_info(pool2)
                '''

                # 替换第二层
                net = inception_module(pool1)
                net = tf.layers.batch_normalization(net, training=is_training)

                # alexnet第三层
                conv3 = slim.conv2d(net, num_outputs=128, scope='conv3')
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

