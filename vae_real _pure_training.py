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
out_path = "/media/sinnis/Seagate Expansion Drive/machine_learning_data/G_template/6X_test14/out"
test_path = "/media/sinnis/Seagate Expansion Drive/machine_learning_data/G_template/6X_test14/test"

if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

if_train = False
if_decompress = True




mb_size = 64
z_dim = 10
X_dim = 128 * 128 * 1
h_dim = 128

image_size = 128
c = 0
lr = 1e-3

qx_dim = 32 * 3
pz_dim = 32 * 3


def plot(samples, name, path):
    batchSz = np.shape(samples)[0]
    nRows = np.ceil(batchSz / 8)
    nCols = min(8, batchSz)
    save_images(samples, [nRows, nCols],
                os.path.join(path, name))

gl = gl_ob(image_size, image_size, mb_size, mode='random_xyz_with_seed')
test_x, _ = gl.draw_ob()
X_test_mb = np.reshape(test_x[:mb_size] / 127.5 - 1., (mb_size, image_size, image_size, 1))
plot(X_test_mb, "real for test.png", test_path)
gl.shut_down()

gl.initiate()
gl.mode = 'random_xyz'




def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

q_bns = [batch_norm(name='q_bn{}'.format(i, )) for i in range(1, 5)]

X = tf.placeholder(tf.float32, shape=[None, 128, 128, 1])
z = tf.placeholder(tf.float32, shape=[None, z_dim])


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
z_sample = sample_z(z_mu, z_logvar)

bit, logits = P(z_sample)

bit_samples, X_samples = P(z, True)

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
print(cross_entropy)
# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(cross_entropy+1e-10 * kl_loss)

vars = tf.trainable_variables()
e_vars = [var for var in vars if 'PZ' in var.name]
d_vars = [var for var in vars if 'QX' in var.name]

ae_vars = e_vars + d_vars
for var in ae_vars:
    print var.name

solver = tf.train.AdamOptimizer(0.000005, beta1=0.5).minimize(vae_loss, var_list=ae_vars)

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


    smallest_loss = 1e10
    for it in range(24001, 100000):

        batch_im,_ = gl.draw_ob()

        # for i in range(mb_size):
        #     im = batch_im[i]
        #     scipy.misc.imsave('/media/sinnis/Seagate Expansion Drive/machine_learning_data/G_template/data set/{}.jpg'.format(i+mb_size*(it-1)), im)

        X_mb = np.reshape(batch_im / 127.5 - 1., (mb_size, image_size, image_size, 1))

        _, loss_recon, cross_ent, loss_kl, SumX, Sumbit, loss1 = sess.run([solver, recon_loss, cross_entropy, kl_loss, sum_X1, sum_bit1, loss_1],
                                                               feed_dict={X: X_mb})

        if np.mean(loss_recon) < smallest_loss:
            smallest_loss = np.mean(loss_recon)


        print(
        'Iter: {}   recon_Loss: {:.4}   entropy: {:.4}   kl_loss: {:.4}  sumofX: {:.4} sumofbit: {:.4} correct: {:.4} smallest: {:.4}'.format(
            it,
            np.mean(loss_recon), np.mean(cross_ent), np.mean(loss_kl),
            np.mean(SumX), np.mean(Sumbit),
            np.mean(loss1), smallest_loss))

        if it % 100 == 0:

            samples_t, loss_recon, cross_ent, loss_kl, SumX, Sumbit, loss1 = sess.run(
                [bit, recon_loss, cross_entropy, kl_loss, sum_X1, sum_bit1, loss_1], feed_dict={X: X_test_mb})

            plot(samples_t*2-1, "{}_test.png".format(it), test_path)

            print("test saved in file: %s" % test_path)
            text = '{}      {:.4}  {:.4}       {:.4}      {:.4}  {:.4}  {:.4}'.format(
                    it,
                    np.mean(loss_recon), np.mean(cross_ent), np.mean(loss_kl),
                    np.mean(SumX), np.mean(Sumbit),
                    np.mean(loss1))
            print('\n'+text+'\n')

            with open('./record', 'a') as f:
                if it % 5000 == 0 :
                    f.write('\n---------------------------------------------------------------------------------------'
                            '-------------------------------------------\n' + text)
                else:
                    f.write('\n'+text)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver.save(sess, checkpoint_dir + "/VAE.model")
            print("Model saved in file: %s" % checkpoint_dir)













