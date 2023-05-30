import enum
from random import shuffle
import numpy as np
import tensorflow as tf
from models import perception as pc
from models import dynamics as dy
from dataset import *
import keras.layers as layers
from absl import app
from absl import flags
import loss

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
    for i in range(1, 8):
        true = tf.reduce_sum(truth[..., i] * pred[..., i])
        u = tf.reduce_sum(truth[..., i]) + tf.reduce_sum(pred[..., i]) - true
        mIoU += (true + smooth) / (u + smooth)
    # print("miou:", mIoU / (FLAGS.num_slots - 1))
    return mIoU / (8 - 1)

def mask_seg(mask):
    """
    preprocess the mask to get the K channels mask for K objects
    input: mask (batch_size, height, width)
    output: mask_seg (batch_size,  height, width, K)
    in our case, K = 8
    """
    mask_seg = tf.one_hot(indices=mask, depth = 8, on_value=1.0, off_value=0.0, axis = 0)
    mask_seg = tf.transpose(mask_seg, perm=[1,2,3,0])
    # print("one-hot mask:", mask_seg.shape)
    return mask_seg

def test_vae_step(perception, batch):
        """
        Perform a single test step.
        
        Args:
            perception: perception model
            batch: a batch of data from dataset
        """
        image = batch['image']
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        # print("mask:",batch['mask'].shape)
        mask  = mask_seg(batch['mask'])
        mask = tf.cast(mask, tf.float32)     
        pred_i, pred_mask, logvar, mean, z_sample = perception(image, mask)
        #pred_mask: (batch_size, h, w, num_slots)
        #pred_image.shape:  (batch_size, h, w, 3, num_slots)

        mask_value = tf.tile(tf.expand_dims(pred_mask, axis = -2), [1,1,1,3,1])
        pred_i = pred_i*mask_value
        pred_i = tf.reduce_sum(pred_i, axis = -1)
        print("pred_image:",pred_i.shape)
        print("original_image:",image.shape)           
        
        mIoU  = compute_mIoU(mask, pred_mask)
        print("original mIoU:", mIoU)
        construction_error = loss.l2_loss(pred_i, image)
        print("construction_error2:", construction_error)

def test_vae_step_1(perception, batch):
    """
    Perform a single test step.
    """
    image = batch['image']
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image_b = tf.reshape(image, [-1, 64,64,3])
    # print("mask:",batch['mask'].shape)
    mask = batch['mask']

    mask = tf.reshape(mask, [-1, 64, 64])
    mask  = mask_seg(mask)
    mask = tf.reshape(mask, [-1, 15, 64, 64, 8])
    mask = tf.cast(mask, tf.float32)
    # mask = tf.reshape(mask, [-1, 64, 64, 8])
    # pred_video, pred_mask, logvar, mean, z_sample = perception(image, mask)
    latent_code = pc.video_encoding(perception, image, mask)
    pred_video, pred_mask = pc.video_decoding(perception, latent_code)      
    pred_video = tf.reduce_sum(pred_video, axis = -1)
    # print("pred_mask:", pred_mask.shape)
    # print("mask:", mask.shape)
    
    mask_b = tf.reshape(mask, [-1, 64, 64, 8])
    mIoU  = compute_mIoU(mask, pred_mask)
    print("mIoU1:", mIoU)
    pred_video = tf.reshape(pred_video, [-1, 64, 64, 3])
    image = tf.reshape(image, [-1, 64, 64, 3])
    construction_error = loss.l2_loss(pred_video, image)
    print("construction_error1:", construction_error)
    


batch_size = 4


if __name__ == '__main__':
    
    test_ds_2 = make_freeform_image_dataset(is_train = False)
    test_ds = make_freeform_tfrecord_dataset(is_train = False, shuffle = False)
    # test_ds = make_freeform_tfrecord_dataset(is_train = False)
    test_ds = test_ds.batch(batch_size)
    test_ds_2 = test_ds_2.batch(60)
    #perception model
    perception = pc.ComponentVAE(input_channels = 3, height= 64, width = 64, latent_dim = 16)
    ckpt_e = tf.train.Checkpoint(network=perception)
    ckpt_manager_e = tf.train.CheckpointManager(checkpoint=ckpt_e,
                                              directory="/home/stu4/wlg/PLATO/checkpoint/perception/num_slot_8/slot_size_16_learning_rate_0.0004_",
                                              max_to_keep = 5)
    checkpoint_path = "/home/stu4/wlg/PLATO/checkpoint/perception"
    ckpt_e.restore(ckpt_manager_e.latest_checkpoint)
    #dynamics model
    predictor = dy.InteractionLSTM(8,16, use_camera = False)
    
    #test for performance of perception
    for i, data in enumerate(test_ds):
        test_vae_step_1(perception, data)
        if i > 2:
            break

    
    for i, data in enumerate(test_ds_2):
        test_vae_step(perception, data)
        if i > 2: 
            break
        
    # test for interaction between perception and dynamics
    for i, data in enumerate(test_ds):
        # test_vae_step_1(perception, data)
        image = data['image']
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        # image = tf.reshape(image, [-1, 64, 64, 3])
        mask = data['mask']
        mask = tf.reshape(mask, [-1, 64, 64, 1])
        mask = tf.squeeze(mask, axis = -1)
        mask = mask_seg(mask)
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, [4,15,64,64,8])
        camera_pos = data['camera_pose']
        print("camera_pos:", camera_pos.shape)
        latent_code = pc.video_encoding(perception, image, mask)
        pred_latent_code = predictor(latent_code)
        print("pred_latent_code:", pred_latent_code.shape)
        pred_video, pred_mask = pc.video_decoding(perception, pred_latent_code)
        print("pred_mask:", pred_mask.shape)
        print("pred_video:", pred_video.shape)