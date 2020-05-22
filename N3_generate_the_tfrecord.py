"""

# 把图像数据制作成tfrecord

"""

import tensorflow as tf
import os
from PIL import Image
import random
from tqdm import tqdm


def _int64_feature(label):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))


def _bytes_feature(imgdir):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgdir]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_example_nums(tf_records_filenames):
    nums = 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums


def load_file(imagestxtdir, shuffle=False):
    images = []  # 存储各个集中图像地址的列表
    labels = []
    with open(imagestxtdir) as f:
        lines_list = f.readlines()  # 读取文件列表中所有的行
        if shuffle:
            random.shuffle(lines_list)
        for line in lines_list:
            line_list = line.rstrip().split(' ')  # rstrip函数删除指定字符，这里用的rstrip()因为括号内是空格，所以是删除空白部分
            label = []

            for i in range(1):
                label.append(int(line_list[i + 1]))
            # 这里本质就是要line_list[1]，因为这个部分就是存label的，可以用下面一行直接替代
            # label.append(int(line_list[1]))

            # cur_img_dir=images_base_dir+'/'+line_list[0]
            images.append(line_list[0])
            labels.append(label)
    return images, labels


def create_tf_records(image_base_dir, image_txt_dir, tfrecords_dir,
                      resize_height, resize_width, log, shuffle):
    images_list, labels_list = load_file(image_txt_dir, shuffle)

    # 判断是否存在保存tfrecord文件的路径，如果没有，就创建一个。
    tf_dir, tf_name = os.path.split(tfrecords_dir)
    if not os.path.exists(tf_dir):
        os.makedirs(tf_dir)
    tfrecords_dir = tf_dir + '/' + tf_name
    # print(tfrecords_dir)

    writer = tf.python_io.TFRecordWriter(tfrecords_dir)
    # print('len is :', len(images_list))
    # image_name 这个函数虽然没有用到，但是作用仍十分关键。因为后面的zip要求有两个变量。
    print('\n#######################start to create %s###########################' % tf_name)
    for i, [image_name, single_label_list] in enumerate(zip(images_list, labels_list)):

        cur_image_dir = image_base_dir + '/' + images_list[i]

        if not os.path.exists(cur_image_dir):
            print('the image path is not exists')
            continue

        image = Image.open(cur_image_dir)
        image = image.resize((resize_height, resize_width))
        image_raw = image.tobytes()
        single_label = single_label_list[0]

        if i % log == 0 or i == len(images_list) - 1:
            print('------------processing:%d-th------------' % i)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(single_label)}))
        writer.write(example.SerializeToString())
    print('#######################successfully create %s###########################\n' % tf_name)
    writer.close()


if __name__ == '__main__':
    resize_height = 600
    resize_width = 600
    # shuffle = True
    log = 5

    train_image_dir = 'E:/111project/ship image/train'
    train_txt_dir = 'E:/111project/ship image/train.txt'
    train_records_dir = 'E:/111project/tfrecordss/train.tfrecords'
    create_tf_records(train_image_dir, train_txt_dir, train_records_dir,
                      resize_height, resize_width, log, shuffle=True)
    train_nums = get_example_nums(train_records_dir)
    print('the train records number is:', train_nums)

    validation_image_dir = 'E:/111project/ship image/validation'
    validation_txt_dir = 'E:/111project/ship image/validation.txt'
    validation_records_dir = 'E:/111project/tfrecordss/validation.tfrecords'
    create_tf_records(validation_image_dir, validation_txt_dir, validation_records_dir,
                      resize_height, resize_width, log, shuffle=True)
    validation_nums = get_example_nums(validation_records_dir)
    print('the validation records number is:', validation_nums)

    test_image_dir = 'E:/111project/ship image/test'
    test_txt_dir = 'E:/111project/ship image/test.txt'
    test_records_dir = 'E:/111project/tfrecordss/test.tfrecords'
    create_tf_records(test_image_dir, test_txt_dir, test_records_dir,
                      resize_height, resize_width, log, shuffle=False)
    test_nums = get_example_nums(test_records_dir)
    print('the test records number is:', test_nums)
