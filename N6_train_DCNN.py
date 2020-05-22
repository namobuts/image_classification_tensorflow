from N4_using_dataset_to_read_the_tfrecord import *
from N0_set_config import *
from tqdm import tqdm
# from N5_1_AlexNet_with_batchnorm import *
from N5_2_slim_alexnet import *
# from N5_3_slim_vgg16 import *


def main():
    train_file_name = 'E:/111project/tfrecord/train.tfrecords'
    validation_file_name = 'E:/111project/tfrecord/validation.tfrecords'

    train_data = create_dataset(train_file_name, batch_size=batch_size,
                                resize_height=resize_height, resize_width=resize_width, num_class=num_class)
    validation_data = create_dataset(validation_file_name, batch_size=batch_size,
                                     resize_height=resize_height, resize_width=resize_width, num_class=num_class)

    train_data = train_data.repeat()
    validation_data = validation_data.repeat()

    train_iterator = train_data.make_one_shot_iterator()
    val_iterator = validation_data.make_one_shot_iterator()

    train_images, train_labels = train_iterator.get_next()
    val_images, val_labels = val_iterator.get_next()

    # x = tf.placeholder(tf.float32, shape=[None, resize_height, resize_width, 3], name='x')
    # y = tf.placeholder(tf.int32, shape=[None, num_class], name='y')
    # keep_prob = tf.placeholder(tf.float32)
    # is_training = tf.placeholder(tf.bool)

    fc3 = inference(x, num_class, keep_prob, is_training)

    with tf.name_scope('learning_rate'):
        # global_ = tf.Variable(tf.constant(0))
        global_ = tf.placeholder(tf.int32)
        lr = tf.train.exponential_decay(learning_rate, global_,
                                        decay_step, decay_rate, staircase=True)

    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=y))

    with tf.name_scope('optimizer'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('accuracy', accuracy)
    # tf.summary.scalar('learning_rate', lr)
    merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=3)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(train_tensorboard_path, sess.graph)
        val_writer = tf.summary.FileWriter(val_tensorboard_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        max_acc = 0

        for i in tqdm(range(iteration)):
            print('\n')
            print('\niteration: {}'.format(i + 1))
            # train_batch_images, train_batch_labels = sess.run([train_images, train_labels])
            # val_batch_images, val_batch_labels = sess.run([val_images, val_labels])

            # if (i+1) % 10 == 0:
            #     print('\niteration: {}'.format(i + 1))
            try:
                train_batch_images, train_batch_labels = sess.run([train_images, train_labels])
                train_loss, lr1, _, train_acc = sess.run([loss_op, lr, train_op, accuracy],
                                                         feed_dict={x: train_batch_images,
                                                                    y: train_batch_labels,
                                                                    keep_prob: drop_rate,
                                                                    is_training: True,
                                                                    global_: i
                                                                    })

                val_batch_images, val_batch_labels = sess.run([val_images, val_labels])
                val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={x: val_batch_images,
                                                                             y: val_batch_labels,
                                                                             keep_prob: 1.0,
                                                                             is_training: False,
                                                                             global_: i
                                                                             })
                if i % 50 == 0:

                    print('lr is : {}'.format(lr1))
                    print("train loss: %.6f, train acc:%.6f" % (train_loss, train_acc))
                    s = sess.run(merged_summary, feed_dict={x: train_batch_images,
                                                            y: train_batch_labels,
                                                            keep_prob: drop_rate,
                                                            is_training: True,
                                                            global_: i
                                                            })
                    train_writer.add_summary(summary=s, global_step=i)

                    print("val loss: %.6f, val acc: %.6f" % (val_loss, val_acc))
                    print('\n')
                    t = sess.run(merged_summary, feed_dict={x: val_batch_images,
                                                            y: val_batch_labels,
                                                            keep_prob: 1.0,
                                                            is_training: False,
                                                            global_: i
                                                            })
                    val_writer.add_summary(summary=t, global_step=i)

                if val_acc >= max_acc:
                    max_acc = val_acc
                    saver.save(sess, model_save_path + '-' + 'val_acc-%.4f' % max_acc, global_step=i)

            except tf.errors.OutOfRangeError:
                break

        print('\n********it is the end********\n')
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    x = tf.placeholder(tf.float32, shape=[None, resize_height, resize_width, 3], name='x')
    y = tf.placeholder(tf.int32, shape=[None, num_class], name='y')
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    train_tensorboard_path = 'log/slim_alexnet/train/'

    val_tensorboard_path = 'log/slim_alexnet/val/'

    model_save_path = 'model/slim_alexnet/alexnet.ckpt'

    main()
