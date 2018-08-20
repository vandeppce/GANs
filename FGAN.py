import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('data/MNIST/')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Parameters

batch_size = 64
epochs = 200
samples = 16

img_size = data.train.images.shape[1]
noise_size = 100

learning_rate = 0.00005
dropout_rate = 0.2

d_units = 128
g_units = 128

real_img = tf.placeholder(dtype=tf.float32, shape=[None, img_size], name='real_img')
noise_img = tf.placeholder(dtype=tf.float32, shape=[None, noise_size], name='noise_img')

# 可视化函数
def plot_images(images):

    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape((28, 28)), cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# MLP
# discriminator
def build_discriminator(img, d_units, reuse=False):

    with tf.variable_scope('discriminator', reuse=reuse):
        # hidden layer
        layer1 = tf.layers.dense(inputs=img, units=d_units, activation=tf.nn.relu)

        # output layer
        layer2 = tf.layers.dense(inputs=layer1, units=1)

        output = tf.nn.sigmoid(layer2)
        return layer2, output

# generator
def build_generator(img, g_units, output_dim, reuse=False):

    with tf.variable_scope('generator', reuse=reuse):
        # hidden layer
        layer1 = tf.layers.dense(inputs=img, units=g_units, activation=tf.nn.relu)
        # layer1 = tf.layers.dropout(inputs=layer1, rate=dropout_rate)

        # output layer
        layer2 = tf.layers.dense(inputs=layer1, units=output_dim)

        output = tf.nn.tanh(layer2)
        return layer2, output

def f_star(inputs):
    # Using Pearson chi squared
    return 0.25 * tf.square(inputs) + inputs

def build_model():

    # 生成器
    g_outputs_logits, g_outputs = build_generator(img=noise_img, g_units=g_units, output_dim=img_size, reuse=False)

    # 判别器
    d_outputs_real_logits, d_outputs_real = build_discriminator(img=real_img, d_units=d_units, reuse=False)
    d_outputs_fake_logits, d_outputs_fake = build_discriminator(img=g_outputs, d_units=d_units, reuse=True)      # 共享参数

    # 最大化discriminator价值函数
    d_value_real = tf.reduce_mean(d_outputs_real)
    d_value_fake = tf.reduce_mean(f_star(d_outputs_fake))
    d_value = -(d_value_real - d_value_fake)

    # 最小化generator价值函数
    # g_value = tf.reduce_mean(tf.ones_like(d_outputs_fake) - d_outputs_fake)
    g_value = -tf.reduce_mean(f_star(d_outputs_fake))

    # 优化
    train_vars = tf.trainable_variables()

    # generator参数
    g_vars = [var for var in train_vars if var.name.startswith('generator')]

    # discriminator参数
    d_vars = [var for var in train_vars if var.name.startswith('discriminator')]

    # 优化器
    g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=g_value, var_list=g_vars)
    d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=d_value, var_list=d_vars)

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Every epochs
        for i in range(epochs):
            # Every batches
            for batch in range(data.train.images.shape[0] // batch_size):
                batch_x, batch_y = data.train.next_batch(batch_size)

                batch_x = batch_x * 2. - 1.
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

                # 优化discriminator
                _ = sess.run(d_opt, feed_dict={real_img: batch_x, noise_img: batch_noise})
                _ = sess.run(g_opt, feed_dict={noise_img: batch_noise})

            # Calculate loss after every epoch
            d_value_train = sess.run(d_value, feed_dict={real_img: batch_x, noise_img: batch_noise})
            d_value_real_train = sess.run(d_value_real, feed_dict={real_img: batch_x, noise_img: batch_noise})
            d_value_fake_train = sess.run(d_value_fake, feed_dict={real_img: batch_x, noise_img: batch_noise})
            g_value_train = sess.run(g_value, feed_dict={noise_img: batch_noise})

            print("Epoch {}/{}...".format(i + 1, epochs),
                  "Discriminator value: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(d_value_train, d_value_real_train, d_value_fake_train),
                  "Generator Loss: {:.4f}".format(g_value_train))

            # 抽取样本观察
            sample_noise = np.random.uniform(-1, 1, size=(samples, noise_size))
            _, gen_samples = sess.run(build_generator(noise_img, g_units, img_size, reuse=True),
                                   feed_dict={noise_img: sample_noise})

            if i % 50 == 0 or i == 199:
                plot_images(gen_samples)

if __name__ == "__main__":
    build_model()