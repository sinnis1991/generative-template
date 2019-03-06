import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import itertools
from glob import glob
from gl_model import gl_ob
from ops import *
from utils import *

checkpoint_dir = "/media/sinnis/Seagate Expansion Drive/machine_learning_data/G_template/6X_test14/checkpoint"
out_path = "./z_complete_data"

if not os.path.exists(out_path):
    os.makedirs(out_path)

if_train = False
if_decompress = True

mb_size = 64
z_dim = 10
X_dim = 128 * 128 * 1
h_dim = 128

image_size = 128
c = 0
lr = 1e-3
batch_shape = [mb_size,image_size,image_size,1]

qx_dim = 32 * 3
pz_dim = 32 * 3


def plot(samples, name, path):
    batchSz = np.shape(samples)[0]
    nRows = np.ceil(batchSz / 8)
    nCols = min(8, batchSz)
    save_images(samples, [nRows, nCols],
                os.path.join(path, name))

gl = gl_ob(image_size, image_size, mb_size, mode='random_xyz')

def noisy(noise_typ,image,amount_s=0.5):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.01
        amount = amount_s
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

# noisy_im = np.zeros(np.shape(test_x))
#
# for i in range(mb_size):
#     noisy_im[i] = noisy("s&p",test_x[i],0)
#
# X_test_mb = np.reshape(noisy_im[:mb_size] / 127.5 - 1., (mb_size, image_size, image_size, 1))
# plot(X_test_mb, "real for test.png", out_path)
# gl.shut_down()



# gl.initiate()
# gl.mode = 'random_xyz'
#
# init_x, _ = gl.draw_ob()
# X_init_mb = np.reshape(init_x[:mb_size] / 127.5 - 1., (mb_size, image_size, image_size, 1))
# gl.shut_down()





def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

q_bns = [batch_norm(name='q_bn{}'.format(i, )) for i in range(1, 5)]

X = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])

z = tf.get_variable('z_latent', [mb_size, z_dim], initializer=tf.truncated_normal_initializer(mean=0,stddev=0.5))


def Q(X, reuse=False):
    # X dim:128 x 128 x 1
    with tf.variable_scope("QX") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = tf.nn.relu(conv2d(X, qx_dim * 2, name='q_h0_conv'))  # 64 x 64 x qx_dim
        h1 = tf.nn.relu(q_bns[0](conv2d(h0, qx_dim * 2, name='q_h1_conv'), True))  # 32 x 32 x qx_dim*2
        h2 = tf.nn.relu(q_bns[1](conv2d(h1, qx_dim * 4, name='q_h2_conv'), True))  # 16 x 16 x qx_dim*4
        h3 = tf.nn.relu(q_bns[2](conv2d(h2, qx_dim * 8, name='q_h3_conv'), True))  # 8 x 8 x qx_dim*8
        h4 = tf.nn.relu(q_bns[3](conv2d(h3, qx_dim * 16, name='q_h4_conv'), True))  # 4 x 4 x qx_dim*16
        z_mu = linear(tf.reshape(h4, [-1, 4 * 4 * qx_dim * 16]), z_dim,
                      name='q_h4_linear_mu')  # 4 x 4 x qx_dim x 16 = 8192
        z_logvar = linear(tf.reshape(h4, [-1, 4 * 4 * qx_dim * 16]), z_dim, name='q_h4_linear_logvar')
        return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

log_size = int(math.log(image_size) / math.log(2))
p_bns = [batch_norm(name='p_bn{}'.format(i, )) for i in range(log_size - 1)]


def P(z, reuse=False):
    with tf.variable_scope("PZ") as scope:
        if reuse:
            scope.reuse_variables()

        z_, h0_w, h0_b = linear(z, pz_dim * 16 * 4 * 4, 'p_h0_linear', with_w=True)

        hs0 = tf.reshape(z_, [-1, 4, 4, pz_dim * 16])
        hs0 = tf.nn.relu(p_bns[0](hs0, True))

        hs1, _, _ = conv2d_transpose(hs0, [mb_size, 8, 8, pz_dim * 8],
                                     name="p_h1", with_w=True)
        hs1 = tf.nn.relu(p_bns[1](hs1, True))

        hs2, _, _ = conv2d_transpose(hs1, [mb_size, 16, 16, pz_dim * 4],
                                     name="p_h2", with_w=True)
        hs2 = tf.nn.relu(p_bns[2](hs2, True))

        hs3, _, _ = conv2d_transpose(hs2, [mb_size, 32, 32, pz_dim * 2],
                                     name="p_h3", with_w=True)
        hs3 = tf.nn.relu(p_bns[3](hs3, True))

        hs4, _, _ = conv2d_transpose(hs3, [mb_size, 64, 64, pz_dim * 2],
                                     name="p_h4", with_w=True)
        hs4 = tf.nn.relu(p_bns[4](hs4, True))

        hs5, _, _ = conv2d_transpose(hs4, [mb_size, 128, 128, 1],
                                     name="p_h5", with_w=True)

        return tf.nn.sigmoid(hs5), hs5


# =============================== TRAINING ====================================
z_mu, z_logvar = Q(X)
z_samples = sample_z(z_mu, z_logvar)

bit, logits = P(z)

X_1 = (X + 1) / 2.
X_0 = 1 - X_1

bit_1 = bit
bit_0 = 1 - bit

sum_X1 = tf.reduce_sum(X_1, [1, 2, 3])
sum_X0 = X_dim - sum_X1

sum_bit1 = tf.reduce_sum(bit_1, [1, 2, 3])
sum_bit0 = X_dim - sum_bit1

X_bit_1 = X_1 * bit_1
X_bit_0 = X_0 * bit_0

loss_1 = tf.reduce_sum(tf.contrib.layers.flatten(X_bit_1), 1)
loss_0 = tf.reduce_sum(tf.contrib.layers.flatten(X_bit_0), 1)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(X_1 - bit_1)), 1)

X_1_samples = tf.contrib.layers.flatten(X_1)
logits_samples = tf.contrib.layers.flatten(logits)

cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X_1_samples, logits=logits_samples),1)

complete_loss = tf.reduce_mean(recon_loss + cross_entropy)

vars = tf.trainable_variables()
e_vars = [var for var in vars if 'PZ' in var.name]
d_vars = [var for var in vars if 'QX' in var.name]

ae_vars = e_vars + d_vars
for var in ae_vars:
    print var.name

solver = tf.train.AdamOptimizer(10, beta1=0.5).minimize(complete_loss, var_list=[z])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=ae_vars, max_to_keep=1)

    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("""

        ======
        An existing model was found in the checkpoint directory.
        ======

        """)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("""

        ======
        An existing model was not found in the checkpoint directory.
        Initializing a new one.
        ======

        """)

    for n in range(4,1001):

        test_x, _ = gl.draw_ob()
        X_init = np.reshape(test_x / 127.5 - 1., (mb_size, image_size, image_size, 1))
        z_init = sess.run(z_samples, feed_dict={X: X_init})
        init_op = tf.assign(z, z_init)

        sess.run(init_op)

        test_x, test_y = gl.draw_ob()
        # z_test = sess.run(z_samples, feed_dict={X: test_x})
        # init_op2 = tf.assign(z, z_test)
        # sess.run(init_op2)
        # X_test = sess.run(bit)*255
        test_x = np.reshape(test_x, (mb_size, image_size, image_size, 1))


        noisy_im = np.zeros(np.shape(test_x))

        for i in range(mb_size):
            noisy_im[i] = noisy("s&p",test_x[i],0.1)

        X_ori = np.reshape(noisy_im/ 127.5 - 1., (mb_size, image_size, image_size, 1))
        plot(X_ori, "{}_real.png".format(n), out_path)

        Loss_recon = [None]
        Loss_cross = [None]
        Sum_X = [None]
        Sum_bit = [None]
        Loss1 = [None]
        Z_encode = [None]

        for it in range(0, 501):



            _, loss_recon, cross_ent, SumX, Sumbit, loss1 = sess.run([solver, recon_loss, cross_entropy, sum_X1, sum_bit1, loss_1],
                                                                   feed_dict={X: X_ori})

            print(
            'Iter: {}   recon_Loss: {:.4}   entropy: {:.4}  sumofX: {:.4} sumofbit: {:.4} correct: {:.4}'.format(
                it,
                np.mean(loss_recon), np.mean(cross_ent),
                np.mean(SumX), np.mean(Sumbit),
                np.mean(loss1)))

            if it == 500:

                samples_t = sess.run(bit)

                plot(samples_t*2-1, "{}_comp.png".format(n), out_path)

                print("test saved in file: %s" % out_path)

            if it == 500:
                Loss_recon = loss_recon
                Loss_cross = cross_ent
                Sum_X = SumX
                Sum_bit = Sumbit
                Loss1 = loss1
                Z_encode = sess.run(z)

        with open('./z_complete_data/z_data', 'a') as f:
            text = '\nbatch{}\n'.format(n)

            for i in range(mb_size):
                text = text + '{}'.format(Loss_recon[i]) + ' '
                text = text + '{}'.format(Loss_cross[i]) + ' '
                text = text + '{}'.format( Sum_X[i]) + ' '
                text = text + '{}'.format(Sum_bit[i]) + ' '
                text = text + '{}'.format(Loss1[i]) + '           '
                for j in range(6):
                    text = text + '{}'.format(test_y[i][j]) + ' '

                text = text+'        '
                for j in range(z_dim):
                    text = text + '{}'.format(Z_encode[i][j]) + ' '

                if i < mb_size-1:
                    text = text + '\n'
                # text = text+"{} {} {} {}".format(\
                #     Loss_recon[i],
                #     Loss_cross[i]
                # )

            f.write(text)

    # print(Loss_recon, Loss_cross, Sum_X, Sum_bit, Loss1,test_y, Z_encode)

    # z_recover = [None]
    #
    # with open('./z_complete_data/z_data', 'r') as f:
    #
    #     index = 0
    #
    #     while True:
    #         p = f.readline().strip()
    #         if p == 'end':
    #             break
    #         word = p.split()
    #         if len(word) == 21:
    #             z_recover[index] = [float(word[-10]),float(word[-9]),float(word[-8]),float(word[-7]),float(word[-6]),
    #                                 float(word[-5]),float(word[-4]),float(word[-3]),float(word[-2]),float(word[-1])]
    #             if index<63:
    #                 z_recover.append(None)
    #                 index = index + 1
    #
    # print z_recover
    # print len(z_recover)
    #
    # recover_op = tf.assign(z, np.reshape(np.array(z_recover),(mb_size,z_dim)))
    #
    # sess.run(recover_op)
    #
    # X_recover = sess.run(bit)
    # plot(X_recover * 2 - 1, "recover.png", out_path)




