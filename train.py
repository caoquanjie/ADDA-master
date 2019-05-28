import tensorflow as tf
from adda import s_encoder, t_encoder, classifier,discriminator, eval,concat_ad_loss_2
from config import Config
import numpy as np
from utils import fresh_dir
config = Config()

from svhn import load_svhn
from mnist import load_mnist
import logging
import os


def return_dataset(data, scale=False, usps=False, all_use=False):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    # if data == 'usps':
    #     train_image, train_label, \
    #     test_image, test_label = load_usps(all_use=all_use)
    return train_image, train_label, test_image, test_label


def dataset_read(source, target, pixel_norm=True, scale=False, all_use=False):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    usps = False
    if source == 'usps' or target == 'usps':
        usps = True

    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,
                                                                            usps=usps, all_use=all_use)
    train_target, t_label_train, test_target, t_label_test = return_dataset(target, scale=scale, usps=usps,
                                                                            all_use=all_use)
    #normalize with mean value of pixels
    if pixel_norm:
        pixel_mean = np.vstack([train_source, train_target]).mean((0,))
        train_source = (train_source - pixel_mean) / float(255)
        test_source = (test_source - pixel_mean) / float(255)
        train_target = (train_target - pixel_mean) / float(255)
        test_target = (test_target - pixel_mean) / float(255)

    return train_source, test_source, train_target, test_target, s_label_train, s_label_test, t_label_train, t_label_test

train_source, test_source, train_target, test_target, s_label_train, s_label_test, t_label_train, t_label_test = dataset_read(
    source='svhn', target='mnist', pixel_norm=True, scale=True, all_use=False
)


def normalization(img):
    im = rgb2gray(img)  # RGB to greyscale
    return im

def rgb2gray( img):
    return np.expand_dims(np.dot(np.array(img, dtype='float32'), [0.299, 0.587, 0.114]), 3)


source_train_img = np.transpose(train_source,[0,2,3,1])
source_train_lab = s_label_train
source_test_img = np.transpose(test_source,[0,2,3,1])
source_test_lab = s_label_test
target_train_img = np.transpose(train_target,[0,2,3,1])
target_train_lab = t_label_train
target_test_img = np.transpose(test_target,[0,2,3,1])
print(source_train_img.shape)
print(target_test_img.shape)
target_test_lab = t_label_test



# generate a batch data
def fill_feed_dict(img,label,step):
    size = img.shape[0]
    start = (step * config.batch_size) % (size - config.batch_size)
    end = min(start + config.batch_size, size)
    batch_imgs = img[start:end]
    batch_labels = label[start:end]
    return batch_imgs, batch_labels

# set exp results to logs
def set_log_info(name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # True to log file False to print
    logging_file = True
    if logging_file == True:
        hdlr = logging.FileHandler(name)
    else:
        hdlr = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger
logger = set_log_info('adda.log')


if not os.path.exists('source_model'):
    os.mkdir('source_model')
def step1(logdir = './source_model'):
    source_images_ph = tf.placeholder(tf.float32,
                                      [None, config.original_size, config.original_size, config.num_channels])
    labels_ph = tf.placeholder(tf.int64, [None])


    target_images_ph = tf.placeholder(tf.float32,
                                      [None, config.original_size, config.original_size, config.num_channels])

    phase = tf.placeholder(tf.bool)

    # inference classification network
    s_fearture = s_encoder(source_images_ph, reuse=None,trainable=True)
    s_logits = classifier(s_fearture,phase=phase,reuse=None,trainable=True)

    # build loss and create optimizer
    s_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s_logits, labels=labels_ph)
    s_loss_mean = tf.reduce_mean(s_loss)
    pred_labels = tf.argmax(s_logits, 1)

    acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels_ph), dtype=tf.float32))


    var_list = tf.trainable_variables()
    reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list
                         if 'bias' not in v.name]) * 5e-4

    # hybrid loss
    loss = s_loss_mean + reg_loss # `-` for minimize

    global_step = tf.Variable(0, trainable=False)


    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=loss, global_step=global_step)

    # build saver to save best epoch
    all_vars = tf.global_variables()
    var_source = [k for k in all_vars if k.name.startswith("s_encoder")]
    var_cls = [k for k in all_vars if k.name.startswith("classifier")]





    with tf.name_scope("step1"):
        tf.summary.scalar('s_loss', s_loss_mean)
        tf.summary.scalar('train_acc', acc)

    summary = tf.summary.merge_all()

    # start a session
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(config.summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver_source = tf.train.Saver(var_source)
        saver_cls = tf.train.Saver(var_cls)

        for i in range(config.epoch):
            num_batches = target_train_img.shape[0] // config.batch_size

            train_acc = 0

            for j in range(num_batches):

                svhn_train_img, svhn_train_lab = fill_feed_dict(source_train_img, source_train_lab, j)
                mnist_train_img, mnist_train_lab = fill_feed_dict(target_train_img, target_train_lab, j)

                if j and j % 10 == 0:
                    Summary, step = sess.run([summary, global_step], feed_dict={source_images_ph: svhn_train_img,
                                                                                labels_ph: svhn_train_lab,
                                                                                target_images_ph: mnist_train_img,
                                                                                phase: True
                                                                                })
                    writer.add_summary(Summary, step)

                train_result =sess.run([acc, loss, train_op],feed_dict={source_images_ph: svhn_train_img,
                                                                                labels_ph: svhn_train_lab,
                                                                                target_images_ph: mnist_train_img,
                                                                                phase:True

                                                                                })
                train_acc += train_result[0]
            tra_acc = train_acc / num_batches


            test_num = source_test_img.shape[0]
            eval_num = test_num // config.batch_size
            total_accuracy = 0

            for k in range(eval_num):
                mnist_test_img, mnist_test_lab = fill_feed_dict(target_test_img, target_test_lab, k)


                test_result = sess.run([acc, global_step], feed_dict={source_images_ph: mnist_test_img,
                                                                 target_images_ph: mnist_test_img,
                                                                 labels_ph: mnist_test_lab,
                                                                 phase:False


                                                                 })
                total_accuracy += test_result[0]
            te_acc = total_accuracy / eval_num

            # if best_acc < te_acc:
            #     best_acc = te_acc
            saver_source.save(sess, logdir + "/encoder/encoder.ckpt", global_step=i)
            saver_cls.save(sess, logdir + "/classifier/classifier.ckpt", global_step=i)
            logger.info(
                'epoch {}: train_acc = {:3.4f}'.format(
                    i, tra_acc))
            logger.info(
                'epoch {}: test_acc = {:3.4f}'.format(
                    i, te_acc))

def step2(source_dir = './source_model'):
    source_images_ph = tf.placeholder(tf.float32,
                                      [config.batch_size, config.original_size, config.original_size, config.num_channels])
    labels_ph = tf.placeholder(tf.int64, [config.batch_size])

    target_labels_ph = tf.placeholder(tf.int64, [config.batch_size])


    target_images_ph = tf.placeholder(tf.float32,
                                      [config.batch_size, config.original_size, config.original_size, config.num_channels])

    phase = tf.placeholder(tf.bool)

    # inference classification network
    s_feature = s_encoder(source_images_ph, reuse=False,trainable=False)
    s_logits = classifier(s_feature, phase=phase,reuse=False, trainable=False)
    #disc_s = discriminator(s_feature, reuse=False,trainable=True)
    s_acc = eval(s_logits, labels_ph)

    t_feature = t_encoder(target_images_ph, reuse=False,trainable=True)
    t_logits = classifier(t_feature, phase=phase,reuse=True, trainable=False)
    t_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(t_logits, 1), target_labels_ph), tf.float32))
    tf.summary.scalar('t_acc', t_acc)
    #disc_t = discriminator(t_feature, reuse=True,trainable=True)

    g_loss,d_loss = concat_ad_loss_2(s_feature,t_feature)

    global_step = tf.Variable(0, trainable=False)

    # trainable variables for discriminator
    var_dis = tf.trainable_variables('discriminator')
    optim_d = tf.train.AdamOptimizer(1e-4,beta1=0.5, beta2=0.999).minimize(d_loss, var_list=var_dis,global_step=global_step)

    # trainable variables for target network
    var_target = tf.trainable_variables('t_encoder')
    optim_g = tf.train.AdamOptimizer(1e-4,beta1=0.5, beta2=0.999).minimize(g_loss, var_list=var_target,global_step=global_step)

    # the latest checkpoint
    encoder_path = tf.train.latest_checkpoint(source_dir+"/encoder")
    classifier_path = tf.train.latest_checkpoint(source_dir+"/classifier")

    if encoder_path is None:
        raise ValueError("Don't exits in this dir")

    if classifier_path is None:
        raise ValueError("Don't exits in this dir")

    source_var = tf.contrib.framework.list_variables(encoder_path)

    var_s_g = tf.global_variables(scope='s_encoder')
    var_c_g = tf.global_variables(scope='classifier')
    var_t_g = tf.trainable_variables(scope='t_encoder')
	
	# histogram for trainable variables in tensorboard 
    for var in var_s_g:
        tf.summary.histogram(var.op.name,var)
    for var in var_c_g:
        tf.summary.histogram(var.op.name,var)
    for var in var_t_g:
        tf.summary.histogram(var.op.name,var)

	# model param save
    encoder_saver = tf.train.Saver(var_list=var_s_g)
	classifier_saver = tf.train.Saver(var_list=var_c_g)

    # change variables name from s_encoder to t_encoder
    dict_var = {}

    for i in source_var:
		for j in var_t_g:
			if i[0][1:] in j.name[1:]:
                dict_var[i[0]] = j

    fine_turn_saver = tf.train.Saver(var_list=dict_var)
    best_saver = tf.train.Saver(max_to_keep=3)
    
	
	# tensorboard 
	with tf.name_scope("step2"):

      tf.summary.scalar('d_loss', d_loss)
      tf.summary.scalar('g_loss', g_loss)

    summary = tf.summary.merge_all()



    # start a session
    best_acc = 0
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(config.step2_summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        encoder_saver.restore(sess, encoder_path)

        classifier_saver.restore(sess, classifier_path)

        fine_turn_saver.restore(sess, encoder_path)
        saver_dis = tf.train.Saver(var_dis)

        print("model init successfully!")
        for i in range(config.epoch):
            num_batches = target_train_img.shape[0] // config.batch_size

            train_acc = 0
            for j in range(num_batches):

                svhn_train_img, svhn_train_lab = fill_feed_dict(source_train_img, source_train_lab, j)
                mnist_train_img, mnist_train_lab = fill_feed_dict(target_train_img, target_train_lab, j)

                if j and j % 10 == 0:
                    Summary, step = sess.run([summary, global_step], feed_dict={source_images_ph: svhn_train_img,
                                                                                #labels_ph: svhn_train_lab,
                                                                                target_images_ph: mnist_train_img,
                                                                                target_labels_ph: mnist_train_lab,
                                                                                phase:True
                                                                                })
                    writer.add_summary(Summary, step)

                d_result = sess.run([d_loss,  optim_d],
                                        feed_dict={source_images_ph: svhn_train_img,
                                                   #labels_ph: svhn_train_lab,
                                                   target_images_ph: mnist_train_img,
                                                   phase: True


                                                   })

                g_result = sess.run([g_loss, optim_g,s_acc],
                                    feed_dict={source_images_ph: svhn_train_img,
                                               labels_ph: svhn_train_lab,
                                               target_images_ph: mnist_train_img,
                                               phase: True

                                               }
                                     )
                if j and j%100 ==0:
                    logger.info(
                        'step {}/{}: d_loss = {:3.4f}\tg_loss = {:3.4f}\ttrain_acc = {:3.4f}'.format(
                            j, i, d_result[0], g_result[0], g_result[2]))

                train_acc += g_result[2]
            tra_acc = train_acc / num_batches


            test_num = target_test_img.shape[0]
            eval_num = test_num // config.batch_size
            total_accuracy = 0
            for k in range(eval_num):
                mnist_test_img, mnist_test_lab = fill_feed_dict(target_test_img, target_test_lab, k)

                test_result = sess.run([t_acc, global_step], feed_dict={
                                                                      target_images_ph: mnist_test_img,
                                                                      target_labels_ph: mnist_test_lab,
                                                                      phase: False


                                                                      })

                total_accuracy += test_result[0]
            te_acc = total_accuracy / eval_num
            print(te_acc)
                # if best_acc < te_acc:
                #     best_acc = te_acc
                # saver_target.save(sess, logdir +"/target/target.ckpt", global_step= i)
                # saver_dis.save(sess,  logdir +"/discriminater/discriminater.ckpt", global_step= i)
			#
            logger.info(
                'epoch {}: train_acc = {:3.4f}'.format(
                    i, tra_acc))
            logger.info(
                'epoch {}: test_acc = {:3.4f}\tbatch_test_acc = {:3.4f}'.format(
                    i, te_acc, test_result[0]))

