import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import cifar10

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

img_size = cifar10.img_size
img_size_flat = cifar10.img_size_flat
num_channels = cifar10.num_channels
num_classes = cifar10.num_classes

batch_size = 64
noise_size = 100
epochs = 100
n_samples = 16
learning_rate = 0.00005
clipping = 0.01
n_d = 5
leaky_alpha = 0.2

real_img = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, num_channels], name='real_img')
noise_img = tf.placeholder(dtype=tf.float32, shape=[None, noise_size], name='noise_img')

# 导入cifar数据
def load_data():
    class_names = cifar10.load_class_names()
    images_train, cls_train, labels_train = cifar10.load_training_data()
    images_test, cls_test, labels_test = cifar10.load_test_data()

    return class_names, (images_train, cls_train, labels_train), (images_test, cls_test, labels_test)

class_names, data_train, data_test = load_data()
(images_train, cls_train, labels_train) = data_train

# 选出所有的马
cls = (cls_train == 7)
images = images_train[cls]

# 可视化函数
def plot_images(images):

    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        images[i] = (images[i] + 1) / 2.
        ax.imshow(images[i].reshape((img_size, img_size, 3)), cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# DCGAN, no BN
# Discriminator has no sigmoid
def build_discriminator(inputs_img, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):

        # 32 x 32 x 3 to 16 x 16 x 128
        layer1 = tf.layers.conv2d(inputs=inputs_img, filters=128, kernel_size=3, strides=2, padding='same')
        layer1 = tf.nn.leaky_relu(layer1, alpha=leaky_alpha)

        # 16 x 16 x 128 to 8 x 8 x 256
        layer2 = tf.layers.conv2d(inputs=layer1, filters=256, kernel_size=3, strides=2, padding='same')
        layer2 = tf.nn.leaky_relu(layer2, alpha=leaky_alpha)

        # 8 x 8 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(inputs=layer2, filters=512, kernel_size=3, strides=2, padding='same')
        layer3 = tf.nn.leaky_relu(layer3, alpha=leaky_alpha)

        flatten = tf.layers.flatten(inputs=layer3)
        logits = tf.layers.dense(inputs=flatten, units=1)

        output = logits

        return output

def build_generator(inputs_img, output_dim, reuse=False):

    with tf.variable_scope('generator', reuse=reuse):

        # 100 x 1 to 4 x 4 x 512
        layer1 = tf.layers.dense(inputs=inputs_img, units=4 * 4 * 512)
        layer1 = tf.reshape(tensor=layer1, shape=[-1, 4, 4, 512])
        layer1 = tf.nn.relu(layer1)

        # 4 x 4 x 512 to 8 x 8 x 256
        layer2 = tf.layers.conv2d_transpose(inputs=layer1, filters=256, kernel_size=4, strides=2, padding='same')
        layer2 = tf.nn.relu(layer2)

        # 8 x 8 256 to 16 x 16 x 128
        layer3 = tf.layers.conv2d_transpose(inputs=layer2, filters=128, kernel_size=3, strides=2, padding='same')
        layer3 = tf.nn.relu(layer3)

        # 16 x 16 x 128 to 32 x 32 x 3
        logits = tf.layers.conv2d_transpose(inputs=layer3, filters=output_dim, kernel_size=3, strides=2, padding='same')
        output = tf.nn.tanh(logits)

        return output

def get_loss():

    g_output = build_generator(inputs_img=noise_img, output_dim=num_channels, reuse=False)

    d_output_real = build_discriminator(inputs_img=real_img, reuse=False)
    d_output_fake = build_discriminator(inputs_img=g_output, reuse=True)

    d_value = -tf.reduce_mean(d_output_real - d_output_fake)
    g_value = -tf.reduce_mean(d_output_fake)

    return d_value, g_value

def build_model(d_value, g_value):

    train_vars = tf.trainable_variables()

    # generator参数
    g_vars = [var for var in train_vars if var.name.startswith('generator')]

    # discriminator参数
    d_vars = [var for var in train_vars if var.name.startswith('discriminator')]

    # 优化器，不选择Adam和Momentum
    # Discriminator优化器，clipping
    d_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_value)
    clipped_vars_d = [tf.assign(var, tf.clip_by_value(var, -clipping, clipping)) for var in d_vars]

    # merge the clip operations on critic variables
    with tf.control_dependencies([d_opt]):
        d_opt = tf.tuple(clipped_vars_d)

    g_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_value)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("graph/", sess.graph)

        for i in range(epochs):
            for batch_i in range(images.shape[0] // batch_size - 1):

                batch_images = images[batch_i * batch_size: (batch_i + 1) * batch_size]
                batch_images = batch_images * 2 - 1

                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

                for j in range(n_d):
                    _ = sess.run(d_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})

                _ = sess.run(g_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})

            # Calculate loss after every epoch
            d_value_train = sess.run(d_value, feed_dict={real_img: batch_images, noise_img: batch_noise})
            g_value_train = sess.run(g_value, feed_dict={noise_img: batch_noise})

            print("Epoch {}/{}...".format(i + 1, epochs), "Discriminator value: {:.4f}...".format(d_value_train),
                    "Generator Loss: {:.4f}".format(g_value_train))

            # 抽取样本观察
            sample_noise = np.random.uniform(-1, 1, size=(n_samples, noise_size))
            gen_samples = sess.run(build_generator(noise_img, num_channels, reuse=True),
                                      feed_dict={noise_img: sample_noise})

            if i % 50 == 0 or i == 199:
                plot_images(gen_samples)

if __name__ == "__main__":
    d_value, g_value = get_loss()
    build_model(d_value, g_value)
