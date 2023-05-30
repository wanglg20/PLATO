"""
Author: Linge Wang
Introduction: loss function for PLATO module
Latest update: 2022/07/28
"""

import numpy as np
import tensorflow as tf
import keras.layers as layers
import tensorflow_probability as tfp

tfd = tfp.distributions

def compute_mIoU(pred, truth):
    """
    compute Mean Intersection over Union 

    Args:
        pred (..., num_categories): pred segmentation made by model
        truth (..., num_categories): groundtruth segmentation

    Returns:
        mIoU: a number
    """
    mIoU = 0
    smooth = 1e-4
    true = tf.reduce_sum(truth * pred)
    u = tf.reduce_sum(truth) + tf.reduce_sum(pred) - true
    mIoU += (true + smooth) / (u + smooth)
    # print("miou:", mIoU / (FLAGS.num_slots - 1))
    return mIoU 

def l2_loss(pred, ground_truth):
    """
    compute l2 loss over pred and ground_truth 

    Args:
        pred: pred image        (in our case: reconstructed images / latent code)
        truth: groundtruth image

    Returns:
        l2_loss: a number
    """
    return tf.reduce_mean(tf.math.squared_difference(pred, ground_truth))


def video_reconstuction_loss(pred, ground_truth):
    """
    compute video reconstruction loss

    Args:
        pred (batch_size, num_frames, w, h, 3): pred video
        ground_truth (batch_size, num_frames, w, h, 3): original video
    returns:
        loss (batch_size,)
    """
    loss = tf.math.squared_difference(pred, ground_truth)
    loss = tf.reduce_mean(loss, axis =(1,2,3,4))
    return loss

def reconstuction_loss(pred, ground_truth):
    """
    compute video reconstruction loss

    Args:
        pred (batch_size, num_frames, w, h, 3, num_slot): pred video
        ground_truth (batch_size, num_frames, w, h, 3, num_slot): original video
    returns:
        loss (batch_size,)
    """
    loss = tf.math.squared_difference(pred, ground_truth)
    loss = tf.reduce_mean(loss, axis =(1,2,3,4,5))
    return loss

def log_normal_pdf(sample, mean, logvar):
    """
    compute log_probability given the mean, logvar of the Gaussian distribution and our sample

    Args:
        sample: sample data from the distribution
        mean: mean of the Gaussian distribution
        logvar: logvar of the Gaussian distribution
        note that sample, mean, logvar have the same shape
    Returns:
        log_p: log_probability
    """
    log2pi = tf.math.log(2. * np.pi)
    # raxis = [1,2,3,4]
    # return tf.reduce_sum(
    #     -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
    #     axis=raxis)
    return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)



def perception_loss(mean, logvar, pred_x, ground_truth_x,pred_m, ground_truth_m, beta = 0.1, gamma = 10, scale = 0.05):
    """loss for perception model

    Args:
        mean : mean of the latent code      (batch_size, latent_dim, num_slots)
        logvar : logvar of the latent code  (batch_size, latent_dim, num_slots)
        pred_x : reconstructed object       (batch_size, w, h, c, num_slots)  
        ground_truth_x : original image     (batch_size, w, h, c)
        pred_m: reconstructed mask          (batch_size, w, h, num_slots)
        ground_truth_m: mask for object i   (batch_size, w, h, num_slots)
        beta: hyperparameter for KL_z
        gamma: hyperparameter for KL_m
        scale: hyperparameter for scale of the pred distribution
    """
    # print("pred_x shape: ", pred_x.shape)
    # print("pred_m shape: ", pred_m.shape)
    # print("mean", mean)
    # print("logvar", logvar)
    # print("pred_m", pred_m[0][0][0])
    # print("ground_truth_m", ground_truth_m[0][0][0])
    b,w,h,c,K = pred_x.shape
    latent_dim = mean.shape[1]
    
    ground_truth_x = tf.expand_dims(ground_truth_x, axis = -1)
    ground_truth_x = tf.tile(ground_truth_x, [1,1,1,1,K])           #ground_truth_x: (batch_size, w, h, c, K)
    mask_b = tf.expand_dims(ground_truth_m, axis = -2)
    mask_b = tf.tile(mask_b, [1,1,1,c,1])                           #mask_b: (batch_size, w, h, c, K)
    ground_truth_x *= mask_b
    
    dist = tfd.Normal(loc=pred_x, scale=scale)
    likelihood_x = dist.log_prob(ground_truth_x)
    # likelihood_x = log_normal_pdf(ground_truth_x, pred_x, scale)
    likelihood_x *= mask_b
    likelihood_x = tf.reduce_sum(likelihood_x, axis = [1,2,3,4])
    
    # print(likelihood_x.shape)
    mean = tf.reshape(mean, [b,-1,latent_dim])
    logvar = tf.reshape(logvar, [b,-1,latent_dim])
    stddev = tf.exp(0.5 * logvar)
    post_dist_z = tfd.Normal(loc=mean, scale=stddev)
    prior_dist_z = tfd.Normal(loc=0, scale=1)
    KL_z = tfd.kl_divergence(post_dist_z, prior_dist_z)
    # print("KL_z shape: ", KL_z.shape)
    KL_z = tf.reduce_sum(KL_z, axis = [1,2])
    
    smooth = 1e-4
    dist_c = tfp.distributions.Categorical(probs= pred_m + smooth)
    dist_c_truth = tfp.distributions.Categorical(probs= ground_truth_m + smooth)
    KL_c = tfd.kl_divergence(dist_c, dist_c_truth)
    # print("KL_c shape: ", KL_c.shape)
    KL_c = tf.reduce_sum(KL_c,axis=[1,2])
    
    # print("value for KL_c: ", tf.reduce_mean(KL_c))
    # print("value for KL_z: ", tf.reduce_mean(KL_z))
    # print("value for likelihood_x: ", tf.reduce_mean(likelihood_x))
    return - likelihood_x + beta*KL_z + gamma*KL_c


def relative_accuracy(loss_c, loss_s):
    """
    compute relative accuracy

    Args:
        loss_c (num_examples_c, ): surprise loss for possible video
        loss_s (num_examples_s, ): surprise loss for impossible video
    """
    num_c = loss_c.shape[-1]
    num_s = loss_s.shape[-1]
    loss_c = tf.expand_dims(loss_c, axis = 0)           #(1, num_examples_c)
    loss_s = tf.expand_dims(loss_s, axis = -1)          #(num_examples_s, 1)
    loss_c = tf.tile(loss_c, [num_s, 1])                #(num_example_s, num_example_c)
    loss_s = tf.tile(loss_s, [1, num_c])                #(num_example_s, num_example_c)
    
    accuracy = (loss_c <= loss_s )                      #(num_example_s, num_example_c)
    # print("accuracy:", accuracy.shape)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return accuracy