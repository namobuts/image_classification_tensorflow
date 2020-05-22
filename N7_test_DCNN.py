from N4_using_dataset_to_read_the_tfrecord import *
from N0_set_config import *
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import os
# from N5_1_AlexNet_with_batchnorm import *
from N5_2_slim_alexnet import *
# from N5_3_slim_vgg16 import *


def main():

    # 读取数据
    test_file_name = 'E:/111project/tfrecord/test.tfrecords'
    test_data = create_dataset(test_file_name, batch_size=batch_size,
                               resize_height=resize_height, resize_width=resize_width, num_class=num_class)
    test_data = test_data.make_one_shot_iterator()
    test_images, test_labels = test_data.get_next()

    # 定义占位符
    x = tf.placeholder(tf.float32, shape=[None, resize_height, resize_width, 3], name='x')
    y = tf.placeholder(tf.int32, shape=[None, num_class], name='y')
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    global_ = tf.placeholder(tf.int32)

    fc3 = inference(x, num_class, keep_prob, is_training)

    # 开始运行进程
    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 加载模型。这里可以通过tf.train.latest_checkpoint加载保存的最后一个模型，也可以通过模型名字加载指定的模型。
        saver1 = tf.train.Saver()
        # ckpt = tf.train.latest_checkpoint('model/slim_vgg16')
        saver1.restore(sess, model_path)
        print('\n成功加载模型')

        y_true = []
        y_pred = []

        for k in tqdm(range(iteration)):

            try:
                test_batch_images, test_batch_labels = sess.run([test_images, test_labels])
                probability, labels = sess.run([fc3, tf.argmax(fc3, 1)],
                                               feed_dict={x: test_batch_images,
                                                          y: test_batch_labels,
                                                          keep_prob: 1.0,
                                                          is_training: False,
                                                          global_: k
                                                          })

                for predicted_label in labels:

                    y_pred.append(predicted_label)

                for test_label in test_batch_labels:

                    # 将one_hot转成成列表，这样比起运行下面的tf.argmax，更省时间
                    y_true.append(test_label.tolist())

                    # 返回one_hot标签的索引，也即是数字标签，这个方法耗时更多，不建议使用
                    # test_label = sess.run(tf.argmax(test_label))
                    # y_true.append(test_label)

            except tf.errors.OutOfRangeError:
                print('it is the end')
                break

        coord.request_stop()
        coord.join(threads)

        print('\nbefore argmax y_true is :', y_true)

        # 因为数字标签在one_hot格式中是以索引的形式存在，
        # 即label=1的标签，one_hot形式是[0,1,0,....]，其最大值所在的元素序号（索引）就是1。
        # 在这里使用np.argmax的速度比上面tf.argmax的速度快很多
        y_true = [np.argmax(item) for item in y_true]
        print('\nafter argmax y_true is :', y_true)

        # 将真实标签和预测标签保存进txt，方便以后使用

        np.savetxt(txt_path + 'true.txt', y_true, fmt='%d')
        np.savetxt(txt_path + 'pred.txt', y_pred, fmt='%d')
        print('\n成功保存数据')

        category = ['MH', 'ML', 'CB', 'CM', 'CS']

        # 计算混淆矩阵
        conf = tf.confusion_matrix(y_true, y_pred, num_classes=5)
        conf_numpy = conf.eval()
        # print(np.sum(conf_numpy))

        # 计算归一化后的混淆矩阵，即精度
        x = np.array(conf_numpy)
        x = x.astype('float') / x.sum(axis=1)[:, np.newaxis]

        # 开始画热力图
        sns.set()

        conf_df = pd.DataFrame(conf_numpy, index=category, columns=category)  # 将数量矩阵转化为 DataFrame
        x = pd.DataFrame(x, index=category, columns=category)   # 将数量矩阵归一化，并转化为 DataFrame

        # 第一张图，没有归一化的，图像数量的混淆矩阵
        plt.figure(dpi=600)
        sns.heatmap(conf_df, annot=True, cmap='PuBu', fmt='d', vmin=0)
        plt.title('Confusion Matrix', fontsize=15)
        plt.xlabel('Category', fontsize=13)
        plt.ylabel('Accuracy', fontsize=13)
        plt.xticks(fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.savefig(fig_path + 'Confusion_matrix.jpg')

        # 第二张图，归一化后的，精度的混淆矩阵
        plt.figure(dpi=600)
        sns.heatmap(x, annot=True, cmap='PuBu', fmt='.4f', vmin=0, vmax=1)
        plt.title('Confusion Matrix_norm', fontsize=15)
        plt.xlabel('Category', fontsize=13)
        plt.ylabel('Accuracy', fontsize=13)
        plt.xticks(fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.savefig(fig_path + 'norm_confusion_matrix.jpg')

        plt.show()


if __name__ == '__main__':
    model_path = 'model/slim_vgg16/vgg16.ckpt-val_acc-0.9875-32586'

    txt_path = 'label_txt/slim_vgg16/'
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    fig_path = 'exp_fig/slim_vgg16/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    main()

