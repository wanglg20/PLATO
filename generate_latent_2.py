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

def _parse_latent_1_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'camera_pose': tf.io.FixedLenFeature([], tf.string),
        'latent': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(element, data)
    camera_pose = content['camera_pose']
    latent = content['latent']
    image = content['image']
    mask = content['mask']
    # get our 'feature'-- our image -- and reshape it appropriately
    camera_pose = tf.io.parse_tensor(camera_pose, out_type=tf.float32)
    latent = tf.io.parse_tensor(latent, out_type=tf.float32)
    image = tf.io.parse_tensor(image, out_type=tf.uint8)
    mask = tf.io.parse_tensor(mask, out_type=tf.uint8)
    # return {"latent": latent, "image": image, "mask": mask}
    return {"camera_pose": camera_pose, "latent": latent, "image": image, "mask": mask}

def _parse_latent_2_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'camera_pose': tf.io.FixedLenFeature([], tf.string),
        'latent': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'mean': tf.io.FixedLenFeature([], tf.string),
        'logvar': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(element, data)
    camera_pose = content['camera_pose']
    latent = content['latent']
    mean = content['mean']
    logvar = content['logvar']
    
        
    # get our 'feature'-- our image -- and reshape it appropriately
    camera_pose = tf.io.parse_tensor(camera_pose, out_type=tf.float32)
    latent = tf.io.parse_tensor(latent, out_type=tf.float32)
    mean = tf.io.parse_tensor(mean, out_type=tf.float32)
    logvar = tf.io.parse_tensor(logvar, out_type=tf.float32)
    # return {"latent": latent, "image": image, "mask": mask}
    return {"camera_pose": camera_pose, "latent": latent, "mean":mean, "logvar": logvar}


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


def parse_single_latent(camera_pose, latent, image, mask):
    # print(pos.shape)
    #define the dictionary -- the structure -- of our single example
    data = {
        'camera_pose': _bytes_feature(serialize_array(camera_pose)),
        'latent': _bytes_feature(serialize_array(latent)),
        'image': _bytes_feature(serialize_array(image)),
        'mask': _bytes_feature(serialize_array(mask))
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



def write_generator():
    i = 0
    iterator = iter(dataset)
    optional = iterator.get_next_as_optional()
    while optional.has_value().numpy():
        ds = optional.get_value()
        optional = iterator.get_next_as_optional()
        # batch_ds = tf.data.Dataset.from_tensor_slices(ds)
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        writer = tf.io.TFRecordWriter("/home/stu4/wlg/PLATO/data/latent_2/train/image-part-{:0>3}.tfrecord"
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

def write_latents_2_to_tfr_short(data, writer):
    latent_code = data['latent']
    camera_pos = data['camera_pose']
    mean = data['mean']
    logvar = data['logvar']
    (B, t, d) = camera_pos.shape
    for i in range(B):
        current_pos = camera_pos[i]
        # print(current_pos.shape)
        current_latent = latent_code[i]
        out = parse_single_latent_2(camera_pose = current_pos, latent = current_latent, mean = mean[i], logvar=logvar[i])
        writer.write(out.SerializeToString())
        # print("present_video:", i+1, end='\r')
       
       
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
        current_video = tf.expand_dims(image[i], axis=0)
        current_mask = mask[i]
        current_mask = tf.reshape(current_mask, [-1, 64, 64])
        current_mask = mask_seg(current_mask)
        current_mask = tf.reshape(current_mask, [1, 15, 64, 64, 8])
        current_mask = tf.cast(current_mask, tf.float32)
        
        # print("video:",current_video.shape)
        # print("mask:", current_mask.shape)
        latent_code = pc.video_encoding(vae, current_video, current_mask)
        latent_code = tf.squeeze(latent_code, axis =0)
        # print("latent code:",latent_code.shape)
        out = parse_single_latent(camera_pose=data['camera_pose'][i], latent=latent_code, image=data['image'][i], mask=data['mask'][i])
        writer.write(out.SerializeToString())
        print("present_video:", i+1, end='\r') 
        
if __name__ == '__main__':
    dataset = make_latent1_dataset(is_train=True, shuffle=False)
    ITEMS_PER_FILE = 1000
    dataset = dataset.batch(ITEMS_PER_FILE)
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
        write_latents_2_to_tfr_short(data, wri)
        time_left = (time.time() - start_time) * (290 - i)
        print("Time needed: ",
              time_left, "s", "\t",
              "train-part" + "-" + str(i) + ".tfrecord")

    # path = ['/home/stu4/wlg/PLATO/data/latent_3/train/image-part-000.tfrecord']
    # ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
    # ds = ds.map(_parse_tfr_element)
    # example = next(iter(ds))
    # print(example.keys())
    # print(example['latent'].shape)
    # print(example['camera_pose'].shape)
