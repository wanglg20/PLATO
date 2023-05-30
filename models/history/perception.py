"""
Author: Linge Wang
Introduction: perception modules for the PLATO 
Latest update: 2022/08/16 
"""

import numpy as np
import tensorflow as tf
import keras.layers as layers


class Encoder(layers.Layer):
    """CNN Encoder for Component VAE"""

    def __init__(self, input_channels, height, width, latent_dim=16):
        super(Encoder, self).__init__()
        self.input_nc = input_channels
        self.latent_dim = latent_dim
        self.h = height
        self.w = width
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(height, width, input_channels + 1)),
            layers.Conv2D(filters=32,
                          kernel_size=3,
                          strides=2,
                          activation='relu'),
            layers.Conv2D(filters=32,
                          kernel_size=3,
                          strides=2,
                          activation='relu'),
            layers.Conv2D(filters=64,
                          kernel_size=3,
                          strides=2,
                          activation='relu'),
            layers.Conv2D(filters=64,
                          kernel_size=3,
                          strides=2,
                          activation='relu'),
            layers.Flatten(),
            # No activation
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim + latent_dim),
        ])

    def call(self, x_input, x_mask):
        #x_input: videos of shape (batch_size, num_frames, H, W, num_channels)
        #x_mask: masks for video of shape (batch_size, num_frames, H, W, 1)
        x = tf.concat([x_input, x_mask], axis=-1)
        params = self.encoder(x)
        z_mu = params[:, :self.latent_dim]
        z_logvar = params[:, self.latent_dim:]
        return z_mu, z_logvar


class Decoder(layers.Layer):
    """"Spatial Broadcast Decoder for Component VAE"""

    def __init__(
        self,
        height,
        width,
        latent_dim=16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.h = height
        self.w = width
        self.decoder = tf.keras.Sequential([
            # layers.InputLayer(input_shape=(latent_dim + 2,)),
            layers.Conv2DTranspose(filters=32,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   activation='relu'),
            layers.Conv2DTranspose(filters=32,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   activation='relu'),
            layers.Conv2DTranspose(filters=32,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   activation='relu'),
            layers.Conv2DTranspose(filters=32,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   activation='relu'),
            layers.Conv2DTranspose(filters=4,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same')
        ])

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def spatial_broadcast(self, z, h, w):
        """
        spatial broadcast function for latent code
        input: z: (batch_size, latent_dim)
        output:z_sb: (batch_size, latent_dim+2)
        """
        n = z.shape[0]
        broadcast_shape = tf.constant([1, 1, h, w], tf.int32)
        # z_b = z.reshape(n, -1, 1, 1)
        z_b = tf.reshape(z, [n, -1, 1, 1])
        z_b = tf.tile(z_b, broadcast_shape)
        x = tf.linspace(-1, 1, w)
        y = tf.linspace(-1, 1, h)
        #shape:  x_b: (h,w), y_b: (h,w)
        x_b, y_b = tf.meshgrid(x, y)
        #shape:  x_b:(1,h,w,1), y_b:(1,h,w,1)
        x_b = tf.expand_dims(tf.expand_dims(x_b, axis=0), axis=-1)
        y_b = tf.expand_dims(tf.expand_dims(y_b, axis=0), axis=-1)
        #shape:  x_b:(n,h,w,1), y_b:(n,h,w,1)
        x_b = tf.tile(x_b, [n, 1, 1, 1])
        y_b = tf.tile(y_b, [n, 1, 1, 1])
        x_b = tf.cast(x_b, tf.float32)
        y_b = tf.cast(y_b, tf.float32)
        z_b = tf.transpose(z_b, [0, 2, 3, 1])
        z_sb = tf.concat([z_b, x_b, y_b], axis=-1)
        return z_sb

    def call(self, z_mu, z_logvar):
        z_sample = self.reparameterize(z_mu, z_logvar)
        # print("z_sample1.shape: ", z_sample.shape) #(60, 16)
        z_sb = self.spatial_broadcast(z_sample, self.h, self.w)
        preds = self.decode(z_sb)
        pred_video, pred_mask = preds[:, :, :, :3], preds[:, :, :, 3:]
        # pred_mask = tf.nn.sigmoid(pred_mask)
        return pred_video, pred_mask, z_sample


class ComponentVAE(layers.Layer):
    """
    Component variational autoencoder.
    APIs: 
    encode: (x_input, x_mask) -> (z_mu, z_logvar)
    decode: (z_sample) -> (x_pred, x_mask)
    z_sample: get latent code z for present params
    """

    def __init__(
        self,
        input_channels,
        height,
        width,
        latent_dim=16,
        num_slots=8,
    ):
        super(ComponentVAE, self).__init__()
        self.input_nc = input_channels
        self.latent_dim = latent_dim
        self.h = height
        self.w = width
        self.encoder = Encoder(input_channels, height, width, latent_dim)
        self.decoder = Decoder(height, width, latent_dim)

    def encode(self, x_input, x_mask):
        mean, logvar = self.encoder(x_input, x_mask)
        return mean, logvar

    def sample_z(self, mean, logvar):
        z_sample = self.decoder.reparameterize(mean, logvar)
        return z_sample

    def decode(self, z_sample):
        """
        decode single slot of a latent code
        
        Args:
            z_sample: (batch_size, latent_dim)
        """
        # print(z_sample.shape)
        z_sb = self.decoder.spatial_broadcast(z_sample, self.h, self.w)
        # print("z_sb.shape: ", z_sb.shape)
        preds = self.decoder.decode(z_sb)
        pred_video, pred_mask = preds[:, :, :, :3], preds[:, :, :, 3:]
        return pred_video, pred_mask

    def process_single_slot(self, x_input, x_mask):
        """
        Args:
            x_input: (batch_size, h, w, input_nc)
            x_mask: (batch_size, h, w, 1)
            
        returns:
            pred_image.shape:  (batch_size, w, h, 3)
            pred_mask.shape:   (batch_size, w, h, 1)
            logvar.shape:      (batch_size, latent_dim)
            mean.shape:        (batch_size, latent_dim)
            z_sample.shape:    (batch_size, latent_dim)
        """
        mean, logvar = self.encode(x_input, x_mask)
        pred_video, pred_mask, z_sample = self.decoder(mean, logvar)
        return pred_video, pred_mask, mean, logvar, z_sample

    def encode_all_slots(self, x_input, x_mask):
        """
        encode all slots of a image and generate latent code

        Args:
            x_input (batch_size, h, w, input_nc): input image
            x_mask (batch_size, h, w, num_slot): mask for the image

        Returns:
            z_sample: (batch_size, num_slot, latent_dim)
        """
        B, H, W, num_slots = x_mask.shape

        # x_mask = tf.transpose(x_mask, perm= [0, 3, 1, 2])
        # x_mask = tf.reshape(x_mask, [-1, H, W])
        # x_mask = tf.expand_dims(x_mask, -1)   #(B*num_slots, W, H, 1)
        # x_input = tf.tile(x_input, [num_slots, 1, 1, 1])
        # x_input = x_mask * x_input
        # mean, logvar = self.encode(x_input, x_mask)
        # z_sample = self.sample_z(mean, logvar)
        # z_sample = tf.reshape(z_sample, (B, num_slots, -1))

        z_sample_array = []
        for i in range(num_slots):
            x_mask_i = x_mask[:, :, :, i:i + 1]
            x_input_i = x_input * x_mask_i
            mean, logvar = self.encode(x_input_i, x_mask_i)
            z_sample = self.sample_z(mean, logvar)
            z_sample_array.append(z_sample)
        z_sample = tf.stack(z_sample_array, axis=-2)
        return z_sample

    def decode_all_slots(self, z_sample):
        """given the latent code, generate pred_image and pred_mask

        Args:
            z_sample (batch_size, num_slot, latent_dim): latent code

        Returns:
            pred_image: (batch_size, h, w, input_nc, num_slot)
            pred_mask: (batch_size, h, w, 1, num_slot)
            
        """
        num_slots = z_sample.shape[-2]
        pred_video_array = []
        pred_mask_array = []
        # print("z_sample:", z_sample.shape) #(60,8,16)
        for i in range(num_slots):
            z_sample_i = z_sample[:, i, :]
            z_sb = self.decoder.spatial_broadcast(
                z_sample_i, self.h, self.w)  #(batch_size, latent_dim+2)
            preds = self.decoder.decode(z_sb)  #(batch_size, w, h, 4)
            pred_video, pred_mask = preds[:, :, :, :3], preds[:, :, :, 3:]
            pred_video_array.append(pred_video)
            pred_mask_array.append(pred_mask)

        pred_image = tf.stack(pred_video_array, axis=-1)
        pred_mask = tf.concat(pred_mask_array, axis=-1)
        pred_mask = tf.nn.softmax(pred_mask, axis=-1)
        return pred_image, pred_mask

    def call(self, x_input, x_mask):
        """
        Args:
            x_input: (batch_size, h, w, input_nc)
            x_mask:  (batch_size, h, w, num_slots)
            
        returns: 
            pred_image.shape:  (batch_size, h, w, 3, num_slots)
            pred_mask.shape:   (batch_size, h, w, num_slots)
            logvar.shape:      (batch_size, latent_dim, num_slots)
            mean.shape:        (batch_size, latent_dim, num_slots)
            z_sample.shape:    (batch_size, latent_dim, num_slots)
        """
        B, H, W, C = x_input.shape
        num_slots = x_mask.shape[-1]
        
        # version 2:
        x_mask = tf.transpose(x_mask, perm=[0, 3, 1, 2])
        # x_mask = tf.reshape(x_mask, [-1, H, W])
        x_mask = tf.expand_dims(x_mask, -1)
        x_input = tf.expand_dims(x_input, axis=1)
        x_input = tf.tile(x_input, (1, num_slots, 1, 1, 1))
        # x_input = tf.reshape(x_input, (-1, H, W, C))
        x_input = x_mask * x_input
        x_input = tf.reshape(x_input, shape=[-1, H, W, C])
        x_mask = tf.reshape(x_mask, shape=[-1, H, W, 1])
        pred_image, pred_mask, mean, logvar, z_sample = self.process_single_slot(x_input, x_mask)               #

        pred_image = tf.reshape(pred_image, (B, num_slots, H, W, C))
        pred_image = tf.transpose(pred_image, (0, 2, 3, 4, 1))
        pred_mask = tf.reshape(pred_mask, (B, num_slots, H, W, 1))
        pred_mask = tf.transpose(pred_mask, (0, 2, 3, 4, 1))
        pred_mask = tf.reshape(pred_mask, shape=[B, H, W, num_slots])
        mean = tf.reshape(mean, (B, num_slots, -1))
        mean = tf.transpose(mean, (0, 2, 1))
        logvar = tf.reshape(logvar, (B, num_slots, -1))
        logvar = tf.transpose(logvar, (0, 2, 1))
        z_sample = tf.reshape(z_sample, (B, num_slots, -1))
        z_sample = tf.transpose(z_sample, (0, 2, 1))
        pred_mask = tf.nn.softmax(pred_mask, axis=-1)

        # version 1
        # for i in range(num_slots):
        #     x_mask_i = x_mask[:, :, :, i:i+1]
        #     x_input_i = x_input * x_mask_i
        #     pred_image, pred_mask, mean, logvar, z_sample = self.process_single_slot(x_input_i, x_mask_i)
        #     # print(x_mask_i.shape)
        #     pred_image_array.append(pred_image)
        #     pred_mask_array.append(pred_mask)
        #     mean_array.append(mean)
        #     logvar_array.append(logvar)
        #     z_sample_array.append(z_sample)

        # pred_image = tf.stack(pred_image_array, axis = -1)
        # pred_mask_1 = tf.concat(pred_mask_array, axis = -1)
        # pred_mask_1 = tf.nn.softmax(pred_mask_1, axis = -1)
        # logvar = tf.stack(logvar_array, axis = -1)
        # mean = tf.stack(mean_array, axis = -1)
        # z_sample = tf.stack(z_sample_array, axis = -2)
        return pred_image, pred_mask, mean, logvar, z_sample


def build_perception_model(input_channels=3,
                           height=64,
                           width=64,
                           latent_dim=16,
                           num_slot=8,
                           batch_size=6,
                           **kwargs):
    """Build keras model."""
    perception = ComponentVAE(input_channels=input_channels,
                              height=height,
                              width=width,
                              latent_dim=latent_dim)
    image = tf.keras.Input([height, width] + [input_channels], batch_size)
    mask_in = tf.keras.Input([height, width] + [num_slot], batch_size)
    outputs, mask, logvar, mean, z_sample = perception(image, mask_in)
    model = tf.keras.Model(inputs=(image, mask_in),
                           outputs=(outputs, mask, logvar, mean, z_sample))
    return model


def build_encoder(input_channels=3,
                  height=64,
                  width=64,
                  latent_dim=16,
                  num_slot=8,
                  batch_size=6,
                  **kwargs):
    """Build keras model."""
    perception = ComponentVAE(input_channels=input_channels,
                              height=height,
                              width=width,
                              latent_dim=latent_dim)
    image = tf.keras.Input([height, width] + [input_channels], batch_size)
    mask_in = tf.keras.Input([height, width] + [num_slot], batch_size)
    z_sample = perception.encode_all_slots(image, mask_in)
    model = tf.keras.Model(inputs=(image, mask_in), outputs=(z_sample))
    return model


def build_decoder(input_channels=3,
                  height=64,
                  width=64,
                  latent_dim=16,
                  num_slot=8,
                  batch_size=6,
                  **kwargs):
    """Build keras model."""
    perception = ComponentVAE(input_channels=input_channels,
                              height=height,
                              width=width,
                              latent_dim=latent_dim)
    z = tf.keras.Input([num_slot, latent_dim], batch_size)
    pred_video, pred_mask = perception.decode_all_slots(z)
    model = tf.keras.Model(inputs=(z), outputs=(pred_video, pred_mask))
    return model


def video_encoding(model, x_input, x_mask):
    """
    encode video into latent code

    Args:
        model: perception model 
        x_input: (batch_size, t, h, w, input_nc)
        x_mask: (batch_size, t, h, w, num_slots)

    Returns:
        latent_code: (batch_size, t, num_slots, latent_dim)
    """
    B, T, H, W, C = x_input.shape
    num_slot = x_mask.shape[-1]
    image = tf.reshape(x_input, [-1, 64, 64, 3])
    image_mask = tf.reshape(x_mask, [-1, 64, 64, num_slot])
    latent_code = model.encode_all_slots(
        image, image_mask)  # (B*T,num_slot,latent_dim)
    latent_dim = latent_code.shape[-1]
    latent_code = tf.reshape(latent_code, (-1, T, num_slot, latent_dim))
    return latent_code


def video_decoding(model, latent_code):
    """
    decode latent code into video

    Args:
        model: perception model
        latent_code: (batch_size, t, num_slots, latent_dim)

    Returns:
        pred_video: (batch_size, t, h, w, 3, num_slots)
        pred_mask: (batch_size, t, h, w, num_slots)
    """

    B, T, num_slot, latent_dim = latent_code.shape
    latent_code = tf.reshape(latent_code, (-1, num_slot, latent_dim))
    pred_video, pred_mask = model.decode_all_slots(latent_code)
    pred_mask = tf.reshape(pred_mask, (B, T, 64, 64, -1))
    # print("pred_video.shape: ", pred_video.shape) # (B*T, 64, 64, 3, num_slot)
    pred_video = tf.reshape(pred_video, (B, T, 64, 64, 3, 8))
    mask_weight = tf.tile(tf.expand_dims(pred_mask, axis=-2), [1,1,1,1,3,1])
    pred_video = pred_video * mask_weight
    
    # pred_video = tf.reduce_sum(pred_video, axis=-1)
    
    return pred_video, pred_mask


if __name__ == "__main__":

    perception = ComponentVAE(input_channels=3,
                              height=64,
                              width=64,
                              latent_dim=16)

    image = tf.random.normal([6, 64, 64, 3])
    mask = tf.random.normal([6, 64, 64, 8])
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    pred_i, pred_mask, logvar, mean, z_sample = perception(image, mask)

    print("pred_i.shape: ", pred_i.shape)
    print("pred_mask.shape: ", pred_mask.shape)
    print("logvar.shape: ", logvar.shape)
    print("mean.shape: ", mean.shape)
    print("z_sample.shape: ", z_sample.shape)