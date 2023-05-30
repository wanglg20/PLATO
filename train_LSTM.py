"""
Author: Linge Wang
Introduction: training process for perception model of PLATO
Latest update: 2022/08/12
"""
# from turtle import pos
import enum
import numpy as np
import tensorflow as tf
import random
import datetime
import time

from tensorboardX import SummaryWriter
from tensorflow.python.platform import tf_logging as logging
import tensorflow_probability as tfp
tfd = tfp.distributions

from absl import app
from absl import flags

# from data.visualization import *
from dataset import *
from models import perception as pc
from models import dynamics as dy
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
flags.DEFINE_string("model_dy_dir", "./checkpoint/dynamics/num_slot_8/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 80, "Batch size for the model.")
flags.DEFINE_integer("num_frames", 15, "Number of frames in image.")
flags.DEFINE_integer("num_slots", 8, "Number of slots in Slot Attention.")
flags.DEFINE_integer("slot_size", 16, "dimension of slot.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
flags.DEFINE_integer("max_epochs", 500, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 5000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 1000000,
                     "Number of steps for the learning rate decay.")
flags.DEFINE_bool("load_pretrained_weight", True,
                     "Whether to load pretrained weight.")

######################################
###      some useful function      ###
######################################

def mask_seg(mask):
    """
    preprocess the mask to get the K channels mask for K objects
    input: mask (batch_size, height, width)
    output: mask_seg (batch_size,  height, width, K)
    in our case, K = 8
    """
    mask_seg = tf.one_hot(indices=mask, depth = FLAGS.num_slots, on_value=1.0, off_value=0.0, axis = 0)
    mask_seg = tf.transpose(mask_seg, perm=[1,2,3,0])
    # print("one-hot mask:", mask_seg.shape)
    return mask_seg

def compute_mIoU(pred, truth):
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
    
    # parameters initialization
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
    viz = TensorboardViz(logdir='./tensorboard/dynamic/' + hyperpara)
    para = ['continuity', 'solidity', 'directional_inertia', 'unchangeableness', 'object_persistence']
    
    #loading the dataset
    # train_ds = make_freeform_tfrecord_dataset(is_train = True, shuffle = True)
    test_batch_size = 50
    train_ds = make_latent2_dataset(is_train = True, shuffle = True)
    probe_continuity = make_probe_tfrecord_dataset(para[0], shuffle=True)
    probe_continuity = probe_continuity.batch(test_batch_size)
    probe_solidity = make_probe_tfrecord_dataset(para[1], shuffle=True)
    probe_solidity = probe_solidity.batch(test_batch_size)
    probe_directional_inertia = make_probe_tfrecord_dataset(para[2], shuffle=True)
    probe_directional_inertia = probe_directional_inertia.batch(test_batch_size)
    probe_unchangeableness = make_probe_tfrecord_dataset(para[3], shuffle=True)
    probe_unchangeableness = probe_unchangeableness.batch(test_batch_size)
    probe_object_persistence  = make_probe_tfrecord_dataset(para[4], shuffle=True)
    probe_object_persistence = probe_object_persistence.batch(test_batch_size)
    test_ds = [probe_continuity, probe_solidity, probe_directional_inertia, probe_unchangeableness, probe_object_persistence]
    train_ds = train_ds.batch(batch_size)
    
    test_reconstrct_ds = make_latent1_dataset(is_train=False)
    test_reconstrct_ds = test_reconstrct_ds.batch(test_batch_size)
    
    #define the perception model: 
    perception = pc.ComponentVAE(input_channels = 3,
                                           height= resolution[0], width = resolution[1], latent_dim = slot_size)
    # set the checkpoint
    ckpt_e = tf.train.Checkpoint(network=perception)
    ckpt_manager_e = tf.train.CheckpointManager(checkpoint=ckpt_e,
                                              directory="/home/stu4/wlg/PLATO/checkpoint/perception/num_slot_8/slot_size_16_learning_rate_0.0004_",
                                              max_to_keep = 5)
    checkpoint_path = "/home/stu4/wlg/PLATO/checkpoint/perception"
    ckpt_e.restore(ckpt_manager_e.latest_checkpoint)
    
    
    #define the prediction model
    lstm = dy.InteractionLSTM(num_slots, slot_size, use_camera = True)
    optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)
    ckpt = tf.train.Checkpoint(network=lstm,
                                   optimizer=optimizer,
                                   global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=FLAGS.model_dy_dir +
                                              hyperpara,
                                              max_to_keep = 15)
    
    if FLAGS.load_pretrained_weight:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("load from checkpoint: {}".format(ckpt_manager.latest_checkpoint))
    # Define our metrics
    
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.Mean('test_accuracy', dtype=tf.float32)
    # test_relative_surprise = tf.keras.metrics.Mean('test_relative_accuracy', dtype=tf.float32)
    test_relative_accuracy = tf.keras.metrics.Mean('test_relative_accuracy', dtype=tf.float32)
    test_reconstruction_error = tf.keras.metrics.Mean('test_reconstruction_error', dtype=tf.float32)
    
    #relative surprise
    continuity_relative_surprise = tf.keras.metrics.Mean('continuity_relative_surprise', dtype=tf.float32)
    solidity_relative_surprise = tf.keras.metrics.Mean('solidity_relative_surprise', dtype=tf.float32)
    directional_inertia_relative_surprise = tf.keras.metrics.Mean('directional_inertia_relative_surprise', dtype=tf.float32)
    unchangeableness_relative_surprise = tf.keras.metrics.Mean('unchangeableness_relative_surprise', dtype=tf.float32)
    object_persistence_surprise = tf.keras.metrics.Mean('object_persistence_surprise', dtype=tf.float32)
    
    relative_surprise_array = [continuity_relative_surprise, solidity_relative_surprise, directional_inertia_relative_surprise,
                               unchangeableness_relative_surprise, object_persistence_surprise,]
    
    # @tf.function
    def train_lstm_step(lstm, vae, data, optimizer, training = True):
        #preprocess the data
        camera_pos = data['camera_pose']
        # camera_pos = tf.cast(camera_pos, tf.float32)
        latent_code = data['latent']
        mean = data['mean']
        logvar = data['logvar']
        distr = tfp.distributions.Normal(mean, tf.exp(0.5 * logvar))
        latent_code = distr.sample()
        
        # print(loss.l2_loss(latent_code, data['latent']))
        
        with tf.GradientTape() as tape:
            pred_latent_code = lstm(latent_code, camera_pos)
            pred_latent_code = pred_latent_code[:, :num_frames-1, :, :]
            latent_code = latent_code[:, 1:, :, :]
            mean = mean[:, 1:, :, :]
            logvar = logvar[:, 1:, :, :]
            loss_value = loss.l2_loss(pred_latent_code, latent_code)
            # distr = tfp.distributions.Normal(mean, tf.exp(0.5 * logvar))
            # loss_value = - distr.log_prob(pred_latent_code)
            loss_value = tf.reduce_mean(loss_value)
            train_loss.update_state(loss_value)
            
        gradients = tape.gradient(loss_value, lstm.trainable_weights)
        optimizer.apply_gradients(zip(gradients, lstm.trainable_weights))
        
        
    # @tf.function
    def test_lstm_step(lstm, vae, data, idx):
        """Perform a single test step."""
        possible_image = data['possible_image']
        possible_image = tf.cast(possible_image, tf.float32)
        possible_image = possible_image / 255.0
        possible_mask = data['possible_mask']
        # print("possible_mask",possible_mask.shape)      #[batch_size,2,15,64,64]
        B = possible_mask.shape[0]
        possible_mask = tf.reshape(possible_mask, [-1, 64, 64])
        possible_mask = mask_seg(possible_mask)
        # print("mask_seg:",possible_mask.shape)          #[batch_size,2,15,64,64]
        possible_mask = tf.reshape(possible_mask, [B,2,15,64,64,8])
        possible_mask = tf.cast(possible_mask, tf.float32)
        possible_camera_pos = data['possible_camera_pose']
        possible_camera_pos = tf.cast(possible_camera_pos, tf.float32)
        
        impossible_image = data['impossible_image']
        impossible_image = tf.cast(impossible_image, tf.float32)                       
        impossible_image = impossible_image / 255.0
        impossible_mask = data['impossible_mask']
        impossible_mask = tf.reshape(impossible_mask, [-1, 64, 64])
        impossible_mask = mask_seg(impossible_mask)
        impossible_mask = tf.reshape(impossible_mask, [B,2,15,64,64,8])
        impossible_mask = tf.cast(impossible_mask, tf.float32)
        impossible_camera_pos = data['impossible_camera_pose']
        impossible_camera_pos = tf.cast(impossible_camera_pos, tf.float32)
        # p_surprise_array = []
        # imp_surprise_array = []
        # p_surprise = 0
        # imp_surprise = 0
        p_loss = tf.zeros([B,])
        imp_loss = tf.zeros([B,])
        for i in range(2):
            p_image = possible_image[:, i, ...]
            p_mask = possible_mask[:, i, ...]
            p_camera_pos = possible_camera_pos[:, i, ...]
            
            p_latent_code = pc.video_encoding(vae, p_image, p_mask)
            p_pred_latent_code = lstm(p_latent_code, p_camera_pos)
            p_pred_video, p_pred_mask = pc.video_decoding(vae, p_pred_latent_code)            
            p_pred_video = p_pred_video[:, :num_frames-1, :, :, :]
            p_image, _ = pc.video_decoding(vae, p_latent_code)
            # p_image = tf.reduce_sum()
            p_image = p_image[:, 1:, :, :, :]
            # p_pred_video = tf.reduce_sum(p_pred_video, axis = -1)
            p_loss_value = loss.reconstuction_loss(p_pred_video, p_image)               #(batch_size,)
            p_loss += p_loss_value                                                      #(batch_size,)
            # p_surprise_array.append(p_loss_value)
            # p_surprise += tf.reduce_sum(p_loss_value)
            
            
            imp_image = impossible_image[:, i, ...]
            imp_mask = impossible_mask[:, i, ...]
            imp_camera_pos = impossible_camera_pos[:, i, ...]
            
            imp_latent_code = pc.video_encoding(vae, imp_image, imp_mask)
            imp_pred_latent_code = lstm(imp_latent_code, imp_camera_pos)
            imp_pred_video, imp_pred_mask = pc.video_decoding(vae, imp_pred_latent_code)
            # imp_pred_video = tf.reduce_sum(imp_pred_video, axis=-1)
            imp_pred_video = imp_pred_video[:, :num_frames-1, :, :, :]
            
            imp_image, _ = pc.video_decoding(vae, imp_latent_code)
            imp_image = imp_image[:, 1:, :, :, :]

            imp_loss_value = loss.reconstuction_loss(imp_pred_video, imp_image)
            imp_loss += imp_loss_value
            # imp_surprise_array.append(imp_loss_value)
            # imp_surprise += tf.reduce_sum(imp_loss_value)

            # accuracy = p_loss_value < imp_loss_value
            # accuracy = tf.cast(accuracy, tf.float32)
            # test_accuracy.update_state(tf.reduce_mean(accuracy))

        # print(p_loss.shape)
        accuracy = p_loss < imp_loss
        accuracy = tf.cast(accuracy, tf.float32)
        test_accuracy.update_state(tf.reduce_mean(accuracy))
        test_relative_surprise = relative_surprise_array[idx]
        
        difference = - p_loss + imp_loss
        sum = p_loss + imp_loss
        relative = difference / sum
        relative = tf.reduce_mean(relative)
        test_relative_surprise.update_state(relative)
        # test_relative_surprise.update_state(-(p_surprise-imp_surprise) / (p_surprise+imp_surprise))
        
        return p_loss, imp_loss
        # return p_surprise_array, imp_surprise_array
        # print("relative_surprise:", test_relative_surprise.result())
    
    def test_reconstruction_step(lstm, vae, data):
        image = data['image']
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        latent = data['latent']
        camera_pose = data['camera_pose']
        pred_latent = lstm(latent, camera_pose)
        image_recon, _ = pc.video_decoding(vae, pred_latent)
        # image_recon, _ = pc.video_decoding(vae, latent)
        # image_recon = tf.reduce_sum(image_recon, axis=-1)
        image_recon = image_recon[:, :num_frames-1,...]
        
        image, _ = pc.video_decoding(vae, latent)
        image = image[:, 1:, ...]           
        recon_error = loss.reconstuction_loss(image, image_recon)
        test_reconstruction_error.update_state(tf.reduce_mean(recon_error))    
    
    init_epoch = int(global_step)
    print("start training process")
    for epoch in range(init_epoch, max_epochs):
        start = time.time()
        for batch, data in enumerate(train_ds):
            # print("proceeding images: {} in epoch {}".format(batch*batch_size, epoch), end="\r")
            all_step = global_step * 290000 / batch_size + batch
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

            train_lstm_step(lstm, perception, data, optimizer) 
            
            if not (batch + 1) % 300:
                
                    
                time_consume = time.time() - start
                time_left = time_consume / (batch + 1) * (290000 / batch_size - batch - 1)
                log_fn(
                    "Epoch: {}, Train_Loss: {:.6f}, proceeding to images: {}, time left for the epoch: {:.2f} min"
                    .format(global_step.numpy(),
                            train_loss.result(),
                            (batch + 1) * batch_size,
                            time_left / 60))
                train_loss.reset_states()
                
        viz.update('train_loss', epoch,
                           {'scalar': train_loss.result().numpy()})
        
        #test for training metric 
        
        for idx_test_recon, data_recon in enumerate(test_reconstrct_ds):
            # print("present reconstruction test idx:", idx_test_recon, end='\r')
            test_reconstruction_step(lstm, perception, data_recon)
            
        log_fn(
                "Epoch: {}, test_reconstruction_error: {:.6f}"
                .format(global_step.numpy(),
                        test_reconstruction_error.result().numpy(),
                ))
        viz.update('reconstruction_error', epoch,
                           {'scalar': test_reconstruction_error.result().numpy()})
        test_reconstruction_error.reset_states()
        
        for idx in range(5):
            test_ds_i = test_ds[idx]
            test_name = para[idx]

            p_surprise_list = []
            imp_surprise_list = []
            for batch, data2 in enumerate(test_ds_i):
                p_i, imp_i = test_lstm_step(lstm, perception, data2, idx)
                # p_surprise_list.extend(p_i)
                # imp_surprise_list.extend(imp_i)
                p_surprise_list.append(p_i)
                imp_surprise_list.append(imp_i)
                # print("present idx:", batch + 1, end = '\r')
                if batch >= 24:
                    break
            loss_p = tf.concat(p_surprise_list, axis = 0)
            # print("loss p:",loss_p.shape)
            loss_imp = tf.concat(imp_surprise_list, axis = 0)
            # print("loss imp", loss_imp.shape)
            accuracy = loss.relative_accuracy(loss_p, loss_imp)
            test_relative_accuracy.update_state(accuracy) 
            name = para[idx] + '_relative_surprise'
            viz.update(name, epoch,
                       {'scalar': relative_surprise_array[idx].result().numpy()})
            
            log_fn(
                "Epoch: {}, Train_Loss: {:.6f}, \n Relative_surprise:{:.6f}, Eval_accuracy: {:.6f}, Eval_relative_accuracy: {:.6f}, Test on: {}"
                .format(global_step.numpy(),
                        train_loss.result().numpy(),
                        relative_surprise_array[idx].result().numpy(),
                        test_accuracy.result().numpy(),
                        test_relative_accuracy.result().numpy(),
                        test_name,
                ))
            test_loss.reset_states()
            test_accuracy.reset_states()
            test_relative_accuracy.reset_states()
            relative_surprise_array[idx].reset_state()
            
        log_fn("Time: {} for {} epochs".format(
                datetime.timedelta(seconds=time.time() - start),
                epoch + 1))
        train_loss.reset_states()
        global_step.assign_add(1)

        #save the checkpoint
        saved_ckpt = ckpt_manager.save()
        log_fn("Saved checkpoint: {}".format(saved_ckpt))
        
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    app.run(main)