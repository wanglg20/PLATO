"""
Author: Linge Wang
Introduction: training process for perception model of PLATO
Latest update: 2022/08/17
"""
import numpy as np
import tensorflow as tf

import datetime
import time

from tensorboardX import SummaryWriter
from tensorflow.python.platform import tf_logging as logging
# from tensorflow_probability import distributions as tfd
from absl import app
from absl import flags
from dataset import *
from models import perception as pc
import loss 
import utils.segmentation_metrics as segmentation_metrics
from PIL import Image

######################################
### hyperparameters initialization ###
######################################


class TensorboardViz(object):
    def __init__(self, logdir):
        self.logdir = logdir
        self.writter = SummaryWriter(self.logdir)

    def text(self, _text):
        # Enhance line break and convert to code blocks
        _text = _text.replace('\n', '  \n\t')
        self.writter.add_text('Info', _text)

    def update(self, mode, it, eval_dict):
        self.writter.add_scalars(mode, eval_dict, global_step=it)

    def flush(self):
        self.writter.flush()
        
FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "./checkpoint/perception/num_slot_8/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 360,  "Batch size for the model.")
flags.DEFINE_integer("num_frames", 15, "Number of frames in image.")
flags.DEFINE_integer("num_slots", 8, "Number of slots in Slot Attention.")
flags.DEFINE_integer("slot_size", 16, "dimension of slot.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("max_epochs", 100, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 1000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 100000,
                     "Number of steps for the learning rate decay.")
flags.DEFINE_bool("load_pretrained_weight", True,
                     "Whether to load pretrained weight.")

######################################
###      some useful function      ###
######################################


def mask_seg(mask):
    """
    preprocess the mask to get the K channels mask for K objects
    
    Args:
        input: mask (batch_size, height, width)
        
    Returns:
        output: mask_seg (batch_size,  height, width, K)
    in our case, K = 8
    """ 
    mask_seg = tf.one_hot(indices=mask, depth = FLAGS.num_slots, on_value=1.0, off_value=0.0, axis = 0)
    mask_seg = tf.transpose(mask_seg, perm=[1,2,3,0])
    # print("one-hot mask:", mask_seg.shape)
    return mask_seg

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
    classes = FLAGS.num_slots -1
    smooth = 1e-4
    for i in range(1, FLAGS.num_slots):
        true = tf.reduce_sum(truth[..., i] * pred[..., i])
        u = tf.reduce_sum(truth[..., i]) + tf.reduce_sum(pred[..., i]) - true
        mIoU += (true + smooth) / (u + smooth)
    return mIoU / (FLAGS.num_slots - 1)
            
if logging is None:
    # The logging module may have been unloaded when __del__ is called.
    log_fn = print
else:
    log_fn = logging.warning

######################################
###   main function for training   ###
######################################

    
def main(argv):
    del argv
    global_step = tf.Variable(0,
                              trainable=False,
                              name="global_step",
                              dtype=tf.int64)
    batch_size = FLAGS.batch_size
    num_frames = FLAGS.num_frames
    num_slots = FLAGS.num_slots
    # mlp_layers = FLAGS.mlp_layers
    slot_size = FLAGS.slot_size
    # num_iterations = FLAGS.num_iterations
    base_learning_rate = FLAGS.learning_rate
    max_epochs = FLAGS.max_epochs
    warmup_steps = FLAGS.warmup_steps
    decay_rate = FLAGS.decay_rate
    decay_steps = FLAGS.decay_steps
    tf.random.set_seed(FLAGS.seed)
    resolution = (64,64)
    parameters = ["slot_size", "learning_rate"]
    hyperpara = ""
    for key in parameters:
        hyperpara += key + "_"
        hyperpara += str(FLAGS.get_flag_value(key, "None")) + "_"
    viz = TensorboardViz(logdir='./tensorboard/perception/' + hyperpara)

    #loading the dataset
    train_ds = make_freeform_image_dataset(is_train = True, shuffle = True)
    test_ds = make_freeform_image_dataset(is_train = False)
    train_ds = train_ds.batch(batch_size)
    # train_ds = train_ds.shuffle(buffer_size = 1000)
    test_ds = test_ds.batch(batch_size)
    #define the model: 
    perception = pc.ComponentVAE(input_channels = 3, height= 64, width = 64, latent_dim = 16)
    # optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)
    optimizer = tf.keras.optimizers.RMSprop(base_learning_rate)
    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    train_reconstruction_error = tf.keras.metrics.Mean('train_reconstruction_error', dtype=tf.float32)
    test_reconstruction_error = tf.keras.metrics.Mean('test_reconstruction_error', dtype=tf.float32)
    train_mIoU = tf.keras.metrics.Mean('train_mIoU', dtype=tf.float32)
    test_mIoU = tf.keras.metrics.Mean('test_mIoU', dtype=tf.float32)
    # set the checkpoint
    ckpt = tf.train.Checkpoint(network=perception,
                                   optimizer=optimizer,
                                   global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=FLAGS.model_dir +
                                              hyperpara,
                                              max_to_keep = 5)
    #load the pretrained weight
    if FLAGS.load_pretrained_weight:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("load pretrained weight from: {}".format(ckpt_manager.latest_checkpoint))
        
    # @tf.function
    def train_vae_step(perception, batch, optimizer, training = True):
        """
        Perform a single training step.
        
        Args:
            perception: perception model
            batch: a batch of data from dataset
        """
        image = batch['image']
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        mask  = mask_seg(batch['mask'])
        mask = tf.cast(mask, tf.float32)

        with tf.GradientTape() as tape:

            pred_i, pred_mask, logvar, mean, z_sample = perception(image, mask)
            mask_value = tf.tile(tf.expand_dims(pred_mask, axis = -2), [1,1,1,3,1])
            reconstruction_i = pred_i*mask_value
            reconstruction_i = tf.reduce_sum(reconstruction_i, axis = -1)
            reconstruction_error = loss.l2_loss(reconstruction_i, image)
            train_reconstruction_error.update_state(reconstruction_error)
            loss_value = loss.perception_loss(mean, logvar, pred_i, image, pred_mask, mask)
            train_loss.update_state(loss_value)

        gradients = tape.gradient(loss_value, perception.trainable_weights)
        optimizer.apply_gradients(zip(gradients, perception.trainable_weights))
        mIoU  = compute_mIoU(mask, pred_mask)
        train_mIoU.update_state(mIoU)

    # @tf.function
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
        loss_value = loss.perception_loss(mean, logvar, pred_i, image, pred_mask, mask)  
        mask_value = tf.tile(tf.expand_dims(mask, axis = -2), [1,1,1,3,1])
        reconstruction_i = pred_i*mask_value
        reconstruction_i = tf.reduce_sum(reconstruction_i, axis = -1)
        reconstruction_error = loss.l2_loss(reconstruction_i, image)
        test_reconstruction_error.update_state(reconstruction_error)         
        test_loss.update_state(loss_value)
        mIoU  = compute_mIoU(mask, pred_mask)
        test_mIoU.update_state(mIoU)
        # print("test_mIoU:", test_mIoU.result())

    start = time.time()
    init_epoch = int(global_step)
    print("start training process")
    for epoch in range(init_epoch, max_epochs):
        for batch, data in enumerate(train_ds):
            # print("proceeding to image:", batch*batch_size, end = '\r')
            all_step = global_step * 4050000 / batch_size + batch
            # Learning rate warm-up.
            if all_step < warmup_steps:
                learning_rate = base_learning_rate * tf.cast(
                    all_step, tf.float32) / tf.cast(
                        warmup_steps, tf.float32)
            else:
                learning_rate = base_learning_rate
            learning_rate = learning_rate * (decay_rate**(tf.cast(
                all_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
            optimizer.lr = learning_rate
            # test_vae_step(perception, data)
            train_vae_step(perception, data, optimizer)
            
            if not (batch + 1) % 1000:
                log_fn(
                    "Epoch: {}, Train_Loss: {:.6f}, train_mIoU: {:.3f}, train_reconstruction_error:{:.6f}, proceeding to images: {}"
                    .format(global_step.numpy(),
                            train_loss.result().numpy(),
                            train_mIoU.result() * 100.0,
                            train_reconstruction_error.result().numpy(),
                            (batch+1) * batch_size))
                train_loss.reset_states()
                train_mIoU.reset_states()
                train_reconstruction_error.reset_states()
            # if not (batch + 1) % 10000:
            #     saved_ckpt = ckpt_manager.save()
            #     log_fn("Saved checkpoint: {}".format(saved_ckpt))
                
        for batch, data in enumerate(test_ds):
            # eval_value, ari = test_step(data, model)
            test_vae_step(perception, data)
                
        viz.update('vae_train_loss', epoch,
                       {'scalar': train_loss.result().numpy()})
        viz.update('train_mIoU', epoch,
                   {'scalar': train_mIoU.result().numpy()})
        viz.update('test_mIoU', epoch,
                   {'scalar': test_mIoU.result().numpy()})
        viz.update('vae_eval_loss', epoch,
                   {'scalar': test_loss.result().numpy()})
        viz.update('vae_train_reconstruction_error', epoch,
                   {'scalar': train_reconstruction_error.result().numpy()})
        
        log_fn(
            "Epoch: {}, Train_mIoU: {:.3f}, \n Eval_Loss: {:.6f}, Eval_mIoU:  {:.3f}, Eval_reconstruction_error: {:.6f}"
            .format(global_step.numpy(),
                    train_mIoU.result().numpy() * 100.0,
                    test_loss.result().numpy(),
                    test_mIoU.result().numpy() * 100.0,
                    test_reconstruction_error.result().numpy())
            )
        log_fn("Time: {} for {} epochs".format(
                datetime.timedelta(seconds=time.time() - start),
                epoch + 1))
        
        test_reconstruction_error.reset_state()
        train_reconstruction_error.reset_state()
        train_loss.reset_states()
        train_mIoU.reset_states()
        test_loss.reset_states()
        test_mIoU.reset_states()

        global_step.assign_add(1)        
        #save the checkpoint
        saved_ckpt = ckpt_manager.save()
        log_fn("Saved checkpoint: {}".format(saved_ckpt))
        
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    app.run(main)