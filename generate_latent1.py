from os import times
import os
import re
import time
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

from dataset import *
from models import perception as pc

def _parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'camera_pose': tf.io.FixedLenFeature([], tf.string),
        'latent': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'mean': tf.io.FixedLenFeature([], tf.string),
        'logvar': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(element, data)
    camera_pose = content['camera_pose']
    latent = content['latent']
    raw_image = content['image']
    raw_mask = content['mask']
    mean = content['mean']
    logvar = content['logvar']
    
    # get our 'feature'-- our image -- and reshape it appropriately
    camera_pose = tf.io.parse_tensor(camera_pose, out_type=tf.float32)
    latent = tf.io.parse_tensor(latent, out_type=tf.float32)
    mean = tf.io.parse_tensor(mean, out_type=tf.float32)
    logvar = tf.io.parse_tensor(logvar, out_type=tf.float32)
    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    mask = tf.io.parse_tensor(raw_mask, out_type=tf.uint8)
    
    # return {"latent": latent, "image": image, "mask": mask}
    return {"camera_pose": camera_pose, "latent": latent, "logvar": logvar, "mean": mean,  "image": image, "mask": mask}


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_single_latent(camera_pose, latent, image, mask, mean, logvar):
    # print(pos.shape)
    #define the dictionary -- the structure -- of our single example
    data = {
        'camera_pose': _bytes_feature(serialize_array(camera_pose)),
        'latent': _bytes_feature(serialize_array(latent)),
        'mean': _bytes_feature(serialize_array(mean)),
        'logvar': _bytes_feature(serialize_array(logvar)),
        'image': _bytes_feature(serialize_array(image)),
        'mask': _bytes_feature(serialize_array(mask)),
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def parse_single_latent_2(camera_pose, latent, mean, logvar):
    # print(pos.shape)
    #define the dictionary -- the structure -- of our single example
    data = {
        'camera_pose': _bytes_feature(serialize_array(camera_pose)),
        'latent': _bytes_feature(serialize_array(latent)),
        'mean': _bytes_feature(serialize_array(mean)),
        'logvar': _bytes_feature(serialize_array(logvar)),
     }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

# dataset = make_latent1_dataset(is_train=True, shuffle=False)

# dataset = make_freeform_image_dataset(is_train=True)
dataset = make_freeform_tfrecord_dataset(is_train=False)
ITEMS_PER_FILE = 1000
dataset = dataset.batch(ITEMS_PER_FILE)

def write_generator():
    i = 0
    iterator = iter(dataset)
    optional = iterator.get_next_as_optional()
    while optional.has_value().numpy():
        ds = optional.get_value()
        optional = iterator.get_next_as_optional()
        # batch_ds = tf.data.Dataset.from_tensor_slices(ds)
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        writer = tf.io.TFRecordWriter("/home/stu4/wlg/PLATO/data/latent_1/test/image-part-{:0>3}.tfrecord"
            .format(i),
            options=options) #compression_type='GZIP'
        i += 1
        yield ds, writer, i
    return

def mask_seg(mask):
    """
    preprocess the mask to get the K channels mask for K objects
    input: mask (batch_size, height, width)
    output: mask_seg (batch_size,  height, width, K)
    in our case, K = 8
    """
    mask_seg = tf.one_hot(indices=mask, depth = 8, on_value=1.0, off_value=0.0, axis = 0)
    mask_seg = tf.transpose(mask_seg, perm=[1,2,3,0])
    return mask_seg


def write_latents_to_tfr_short(data, writer, vae):
    image = data['image']
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    mask = data['mask']
    camera_pos = data['camera_pose']
    (B, T, W, H, C) = image.shape
    for i in range(B):
        current_pos = camera_pos[i]
        # print(current_pos.shape)
        current_video = image[i]
        current_video = tf.reshape(current_video, [-1, 64, 64, 3])
        current_mask = mask[i]
        current_mask = tf.reshape(current_mask, [-1, 64, 64])
        current_mask = mask_seg(current_mask)
        current_mask = tf.cast(current_mask, tf.float32)
        pred_i, pred_mask, mean, logvar, z_sample = vae(current_video, current_mask)
        mean = tf.transpose(mean, perm=[0,2,1])
        logvar = tf.transpose(logvar, perm=[0,2,1])
        z_sample = tf.transpose(z_sample, perm=[0,2,1])
        # out = parse_single_latent(latent=z_sample, image=data['image'][i], mask=data['mask'][i])
        out = parse_single_latent(camera_pose=data['camera_pose'][i], latent=z_sample, image=data['image'][i], mask=data['mask'][i], mean=mean, logvar=logvar)
        writer.write(out.SerializeToString())
        print("present_video:", i+1, end='\r')
        
        
perception = pc.ComponentVAE(input_channels = 3,
                                           height= 64, width = 64, latent_dim = 16)
# set the checkpoint
ckpt_e = tf.train.Checkpoint(network=perception)
ckpt_manager_e = tf.train.CheckpointManager(checkpoint=ckpt_e,
                                          directory="/home/stu4/wlg/PLATO/checkpoint/perception/num_slot_8/slot_size_16_learning_rate_0.0004_",
                                          max_to_keep = 5)
# checkpoint_path = "/home/stu4/wlg/PLATO/checkpoint/perception"
ckpt_e.restore(ckpt_manager_e.latest_checkpoint)


for data, wri, i in write_generator():
    start_time = time.time()
    write_latents_to_tfr_short(data, wri, perception)
    time_left = (time.time() - start_time) * (300 - i)
    print("Time needed: ",
          time_left, "s", "\t",
          "train-part" + "-" + str(i) + ".tfrecord")