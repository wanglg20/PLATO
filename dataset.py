import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


#@title Dataset Constants
_NUM_PROBE_TFRECORDS = 20
_NUM_FREEFORM_TRAIN_TFRECORDS = 100
_NUM_FREEFORM_TEST_TFRECORDS = 10

_FREEFORM_FEATURES = dict(
     image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),                                                                                                                                             
     mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),                                                                                                                                              
     camera_pose=tf.io.FixedLenFeature(dtype=tf.float32, shape=(15, 6)),                                                                                                                              
)

_PROBE_FEATURES = dict(                                                                                                                                                                                                
    possible_image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    possible_mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    possible_camera_pose=tf.io.FixedLenFeature(dtype=tf.float32,
                                               shape=(2, 15, 6)),
    impossible_image=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    impossible_mask=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    impossible_camera_pose=tf.io.FixedLenFeature(dtype=tf.float32,
                                                 shape=(2, 15, 6)),                                                                                                                            
)                           

#@title Dataset Utilities                                                                                                                                                                       
def _parse_latent_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'latent': tf.io.FixedLenFeature([], tf.string),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(element, data)
    latent = content['latent']
    raw_image = content['raw_image']
    raw_mask = content['mask']
    # get our 'feature'-- our image -- and reshape it appropriately
    latent = tf.io.parse_tensor(latent, out_type=tf.float32)
    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    mask = tf.io.parse_tensor(raw_mask, out_type=tf.uint8)
    return {"latent": latent, "image": image, "mask": mask}

def _parse_latent1_tfr_element(element):
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
  
  
def _parse_latent2_tfr_element(element):
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



def _parse_freeform_row(row):                                                                                                                                                                                             
  row = tf.io.parse_example(row, _FREEFORM_FEATURES)                                                                                                                                                                      
  row['image'] = tf.reshape(tf.io.decode_raw(row['image'], tf.uint8),
                            [15, 64, 64, 3])                                                                                                  

  row['mask'] = tf.reshape(tf.io.decode_raw(row['mask'], tf.uint8),
                           [15, 64, 64])                                                                                                       
  return row                                                                                                                                                                                                     
                                                                                                                                                                                                                 
def _parse_probe_row(row):                                                                                                                                                                                             
  row = tf.io.parse_example(row, _PROBE_FEATURES)                                                                                                                                                                      
  for prefix in ['possible', 'impossible']:                                                                                                                                                                      
    row[f'{prefix}_image'] = tf.reshape(
        tf.io.decode_raw(row[f'{prefix}_image'], tf.uint8),
        [2, 15, 64, 64, 3])                                                                                                  
    row[f'{prefix}_mask'] = tf.reshape(
        tf.io.decode_raw(row[f'{prefix}_mask'], tf.uint8),
        [2, 15, 64, 64])                                                                                                       
  return row                                                                                                                                                                                                     


def _make_tfrecord_paths(dir_name, subdir_name, num_records):
  root = f'gs://physical_concepts/{dir_name}/{subdir_name}/data.tfrecord'
  paths = [f'{root}-{i:05}-of-{num_records:05}' for i in range(num_records)]
  return paths

def make_freeform_tfrecord_dataset(is_train, shuffle=False):
  """Returns a TFRecordDataset for freeform data."""
  if is_train:
    subdir_str = 'train'
    num_records = _NUM_FREEFORM_TRAIN_TFRECORDS
  else:
    subdir_str = 'test'
    num_records = _NUM_FREEFORM_TEST_TFRECORDS

  tfrecord_paths = _make_tfrecord_paths('freeform', subdir_str, num_records)
  ds = tf.data.TFRecordDataset(tfrecord_paths, compression_type='GZIP')
  ds = ds.map(_parse_freeform_row)                        
  if shuffle:
    ds = ds.shuffle(buffer_size=50)                                                                                                                                                                
  return ds

def make_probe_tfrecord_dataset(concept_name, shuffle=False):
  """Returns a TFRecordDataset for probes data."""
  tfrecord_paths = _make_tfrecord_paths('probes', concept_name, 20)
  ds = tf.data.TFRecordDataset(tfrecord_paths, compression_type='GZIP')
  ds = ds.map(_parse_probe_row)
  if shuffle:
    ds = ds.shuffle(buffer_size=20)                                                                                                                                                                                         
  return ds

def _parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    content = tf.io.parse_single_example(element, data)
    raw_image = content['raw_image']
    raw_mask = content['mask']
    # get our 'feature'-- our image -- and reshape it appropriately
    image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    mask = tf.io.parse_tensor(raw_mask, out_type=tf.uint8)
    return {"image": image, "mask": mask}

def make_freeform_image_dataset(is_train, shuffle=False):
  """Returns a dataset of images from freeform data."""
  if is_train:
    subdir = 'train/'
    num = 90
  else: 
    subdir = 'test/'
    num = 5
  file_path = '/home/stu4/wlg/PLATO/data/image/' + subdir
  filename = [
    file_path + "image-part-{:0>3}.tfrecord".format(i) for i in range(num)
  ]
  # ds = tf.data.TFRecordDataset(filename, compression_type='GZIP')
  if shuffle:
      filename = tf.data.Dataset.from_tensor_slices(filename)
      ds = filename.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
      cycle_length = num,
      block_length=1)
      ds = ds.shuffle(60)
  else:
      ds = tf.data.TFRecordDataset(filename, compression_type='GZIP')
  ds = ds.map(_parse_tfr_element)
  return ds


def make_latent1_dataset(is_train, shuffle=False):
  """
  Returns a dataset of latent code generated by perception model from freeform data.
  
  num of examples in total: 300000  
  keys for each batch: ['latent', 'image', 'mask']
  batch['latent']: latent code for the videos        (num_frames, num_slots, latent_dims)        (15, 8, 16)
  batch['mean']: mean of pred guassian distribution  (num_frames, num_slots, latent_dims)        (15, 8, 16)
  batch['logvar']: logvar of guassian distribution   (num_frames, num_slots, latent_dims)        (15, 8, 16)
  batch['image']: original video of the latent code  (num_frames, w, h, num_channels)        (15, 64, 64, 3)
  batch['mask']: mask for original video             (num_frames, w, h)                         (15, 64, 64)
  batch['camera_pose']                               (num_frames, pos_dim)                            (15,6)   
  """
  
  if is_train:
    subdir = 'train/'
    num = 300
  else: 
    subdir = 'test/'
    num = 4
  file_path = '/home/stu4/wlg/PLATO/data/latent_1/' + subdir
  filename = [
    file_path + "image-part-{:0>3}.tfrecord".format(i) for i in range(num)
  ]
  # ds = tf.data.TFRecordDataset(filename, compression_type='GZIP')
  if shuffle:
      filename = tf.data.Dataset.from_tensor_slices(filename)
      ds = filename.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
      cycle_length = num,
      block_length=1)
      ds = ds.shuffle(60)
  else:
      ds = tf.data.TFRecordDataset(filename, compression_type='GZIP')
  ds = ds.map(_parse_latent1_tfr_element)
  return ds

def make_latent2_dataset(is_train, shuffle=False):
  """
  Returns a dataset of latent code generated by perception model from freeform data.
  
  num of examples in total: 290000 
  keys for each batch: ['latent', 'image', 'mask']
  batch['latent']: latent code for the videos        (num_frames, num_slots, latent_dims)        (15, 8, 16)
  batch['mean']: mean of pred guassian distribution  (num_frames, num_slots, latent_dims)        (15, 8, 16)
  batch['logvar']: logvar of guassian distribution   (num_frames, num_slots, latent_dims)        (15, 8, 16)
  batch['camera_pose']                               (num_frames, pos_dim)                            (15,6)   
  """
  
  if is_train:
    subdir = 'train/'
    num = 290
  else: 
    subdir = 'test/'
    num = 4
  file_path = '/home/stu4/wlg/PLATO/data/latent_2/' + subdir
  filename = [
    file_path + "image-part-{:0>3}.tfrecord".format(i) for i in range(num)
  ]
  # ds = tf.data.TFRecordDataset(filename, compression_type='GZIP')
  if shuffle:
      filename = tf.data.Dataset.from_tensor_slices(filename)
      ds = filename.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
      cycle_length = num,
      block_length=1)
      ds = ds.shuffle(60)
  else:
      ds = tf.data.TFRecordDataset(filename, compression_type='GZIP')
  ds = ds.map(_parse_latent2_tfr_element)
  return ds



if __name__ == '__main__':
  ds = make_latent2_dataset(is_train=True, shuffle=False)
  ds = make_freeform_tfrecord_dataset(is_train=True)
  ds=ds.batch(1)
  num = 0
  for epoch in range(1):
    for i, batch in enumerate(ds):
      num += 1
      print(batch.keys())
      # print(batch['latent'].shape)   #(4,15,8,16)
      # print(batch['mean'].shape)
      # print(batch['logvar'].shape)
      # # print(batch['image'].shape)  
      # # print(batch['mask'].shape)
      # print(batch['camera_pose'].shape)
      print('present idx:', i, end='\r')
      break
  print('\n')
  print("num:", num)
  