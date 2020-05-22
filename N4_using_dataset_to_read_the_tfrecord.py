"""

# 使用dataset读取tfrecord

"""
import tensorflow as tf
import matplotlib.pyplot as plt


# 单个record的解析函数
def decode_example(example, resize_height, resize_width, label_nums):
    dics = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    parsed_example = tf.parse_single_example(serialized=example, features=dics)

    tf_image = tf.decode_raw(parsed_example['image_raw'], out_type=tf.uint8)  # 这个其实就是图像的像素模式，之前我们使用矩阵来表示图像

    # 对图像的尺寸进行调整，调整成三通道图像，如果前面制作tfrecord时的图像大小是多少，这里就要写多少。
    # 比如tfrecord的图像是224，这里就写[224,224,3]；如果是600，就是[600, 600, 3]
    tf_image = tf.reshape(tf_image, shape=[224, 224, 3])

    tf_image = tf.image.resize_images(tf_image, (resize_height, resize_width), method=2)

    tf_image = tf.cast(tf_image, tf.float32) * (1. / 255)  # 对图像进行归一化以便保持和原图像有相同的精度

    tf_label = tf.cast(parsed_example['label'], tf.int64)

    tf_label = tf.one_hot(tf_label, label_nums, on_value=1, off_value=0)  # 将label转化成用one_hot编码的格式

    return tf_image, tf_label


# 生成dataset，可以引入其他py文件中。
def create_dataset(tfrecords_file, batch_size, resize_height, resize_width, num_class):
    dataset = tf.data.TFRecordDataset(tfrecords_file)

    # map函数可以让dataset的结构和上面的decode_example一致。
    # lambda个人推测是为了构建上面的decode_example，但是如果decode函数除了example之外没有其他变量，那就不需要用lambda
    # 但是这里的lambda里只有一个x，对应的example，而且这个x不需要外部函数传值，这一点很困惑。
    # 我个人觉得是在tf.data.TFRecordDataset创建了dataset之后，dataset里就包含了serialized对应的值
    # 因此可以函数内部自己传值，不需要外部设置变量再传值。
    dataset = dataset.map(lambda x: decode_example(x, resize_height, resize_width, num_class))

    dataset = dataset.shuffle(2000)

    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == '__main__':

    # 定义可以一次获得多张图像的函数
    def show_image(image_dir):
        plt.imshow(image_dir)
        plt.axis('on')
        plt.show()


    # 检查dataset是否正确生成
    def batch_test(tfrecords_file, batch_size, resize_height, resize_width, num_class):

        dataset = create_dataset(tfrecords_file, batch_size, resize_height, resize_width, num_class)

        # iterator = dataset.make_one_shot_iterator()

        # iterator = tf.data.Iterator.from_structure(dataset.output_types,
        #                                            dataset.output_shapes)

        # iterator1 = iterator.make_initializer(dataset)

        iterator = dataset.make_initializable_iterator()

        batch_images, batch_labels = iterator.get_next()

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # 查看tfrecord的样本数量
            sample_nums = 0
            for record in tf.python_io.tf_record_iterator(tfrecords_file):
                sample_nums += 1
            print('\n#######\n')
            print('this tfrecord file: "{}" has _{}_ samples'.format(tfrecords_file, sample_nums))
            print('\n#######\n')

            for i in range(100):
                # sess.run(iterator1)
                sess.run(iterator.initializer)
                # 如果加上while True，那么上面的i代表着一个epoch,也就是将数据集的全部样本都完成一遍；
                # 如果不加上while True，那么上面的i就代表着一个iteration，也就是使用一个batch_size完成一次。
                # iteration和epoch之间的关系为：1 epoch （所代表的数量）= batch_size * iteration

                # 在这里我选择的是iteration，如果需要epoch可以在添加while True之后再将try-break之间的代码tab一下就行。

                try:
                    # show_image(images[5,:,:,:])  # 代表每一个batch的第三张图片
                    images, labels = sess.run([batch_images, batch_labels])
                    print('{}th, image.shape:{}, type:{}, labels.shape:{}'.format(i + 1, images.shape, images.dtype,
                                                                                  labels.shape))

                except tf.errors.OutOfRangeError:
                    print('\n******\n')
                    print('{}th batch is the final batch, total iteration is: {} '.format(i, i))
                    print('\n******\n')
                    break
            coord.request_stop()
            coord.join(threads)


    tfrecords_file = 'E:/111project/tfrecord/validation.tfrecords'
    resize_height = 100
    resize_width = 100
    num_class = 5

    batch_test(tfrecords_file, 200, resize_height, resize_width, num_class)
