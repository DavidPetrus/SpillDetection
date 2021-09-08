import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

import numpy as np

from stn import spatial_transformer_network as spatial_transform

from absl import flags, app

FLAGS = flags.FLAGS

loss_tracker = tf.keras.metrics.Mean(name="loss")
loc_loss_tracker = tf.keras.metrics.Mean(name="loc_loss")
locnet_tracker = tf.keras.metrics.Mean(name="locnet_frac")
acc_tracker = tf.keras.metrics.BinaryAccuracy()

'''all_tracker = tf.keras.metrics.Mean(name="loss")
puddle_tracker = tf.keras.metrics.Mean(name="loss")
video_tracker = tf.keras.metrics.Mean(name="loss")'''

class PrototypeSim(tf.keras.layers.Layer):
    def __init__(self):
        super(PrototypeSim, self).__init__()
        self.prototypes = tf.Variable(tf.random.normal((320,FLAGS.num_prototypes)))
        #self.prototypes = tf.random.normal((320,FLAGS.num_prototypes))

    def call(self, inputs):
        embds = tf.nn.l2_normalize(inputs, axis=-1)
        prototypes = tf.nn.l2_normalize(tf.stop_gradient(self.prototypes), axis=-1)
        sims = tf.matmul(embds,prototypes)

        return sims

class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)

        self.min_crop_size = FLAGS.min_crop_size
        self.max_crop_size = FLAGS.max_crop_size

        #self.resnet_bb = tf.keras.models.load_model("/home/petrus/pretrained_models/simclrV2_resnet50x2")
        self.resnet_bb = EfficientNetB0(weights="effnet_weights/efficientnetb0_notop.h5", include_top=False, input_shape=(FLAGS.input_size,FLAGS.input_size,3))
        #self.loc_conv1 = tf.keras.layers.Conv2D()
        self.loc_hid = tf.keras.layers.Dense(64,activation='relu')
        self.loc_dropout = layers.Dropout(rate=FLAGS.loc_dropout)
        self.loc_out = tf.keras.layers.Dense(3,activation='sigmoid')

        self.conv_base = EfficientNetB0(weights="effnet_weights/efficientnetb0_notop.h5", include_top=False, input_shape=(224,224,3))
        self.max_pool = layers.GlobalMaxPooling2D(name="gap")
        '''self.proj_head = layers.Dense(320, activation="linear", name="fc_out")
        self.prototype_layer = PrototypeSim()
        
        self.gamma = FLAGS.gamma
        self.top_k = FLAGS.top_k'''

        self.cls_dropout = layers.Dropout(rate=FLAGS.cls_dropout)
        self.final = layers.Dense(1, activation="sigmoid", name="fc_out")

        self.min_spill_frac = FLAGS.min_spill_frac
        self.locnet_aug = FLAGS.locnet_aug
        self.loc_coeff = FLAGS.loc_coeff

        #self.loss = tf.keras.losses.BinaryCrossentropy()

        self.zeros32 = tf.zeros(32,dtype=tf.float32)
        self.zeros20 = tf.zeros(20,dtype=tf.float32)
        self.zeros = tf.zeros(32,dtype=tf.float32)

    def call(self, inputs, training=False):
        x = self.resnet_bb(inputs, training=training)
        x = tf.reduce_mean(x,axis=[1,2])

        x = self.loc_hid(x)
        if training:
            x = self.loc_dropout(x)

        crop_pred = self.loc_out(x)
        crop_size = crop_pred[:,0]*(self.max_crop_size-self.min_crop_size) + self.min_crop_size
        cr_x = crop_pred[:,1]*1.8 - 0.9
        cr_y = crop_pred[:,2]*1.8 - 0.9
        if training:
            crop_size = crop_size + (np.random.random()*self.locnet_aug*2 - self.locnet_aug)
            cr_x = cr_x + (np.random.random()*self.locnet_aug*2 - self.locnet_aug)
            cr_y = cr_y + (np.random.random()*self.locnet_aug*2 - self.locnet_aug)

        theta = tf.stack([crop_size,self.zeros,cr_x,self.zeros,crop_size,cr_y])
        #theta = tf.stack([self.zeros+0.3,self.zeros,self.zeros,self.zeros,self.zeros,self.zeros+0.3])
        theta = tf.transpose(theta)
        crops = spatial_transform(inputs,theta,(224,224))

        if training:
            crops = tf.concat([crops[:-20],tf.stop_gradient(crops[-20:])],axis=0)

        x = self.conv_base(crops, training=training)
        x = self.max_pool(x)
        '''x = self.proj_head(x)

        sims = self.prototype_layer(x)

        return x,sims'''

        if training:
            x = self.cls_dropout(x)

        pred = self.final(x)
        return pred, theta

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        self.zeros = self.zeros32

        X,lab = data
        masks,loc_reg_pts = lab
        y = tf.reduce_max(masks,axis=[1,2])

        with tf.GradientTape() as tape:

            pred,theta = self(X,training=True)

            cls_loss = self.compiled_loss(y,pred)

            mask_crops = spatial_transform(masks[:4], theta[:4])
            spill_frac = tf.reduce_mean(mask_crops,axis=[1,2,3])
            '''t_x = theta[:4,2:3]
            t_y = theta[:4,5:6]
            x_diff = tf.reduce_mean(t_x-loc_reg_pts[:,:,1],axis=1)
            y_diff = tf.reduce_mean(t_y-loc_reg_pts[:,:,0],axis=1)
            loc_loss = x_diff**2 + y_diff**2
            w_loc_loss = tf.reduce_sum(tf.nn.relu(self.min_spill_frac-spill_frac)*loc_loss)'''
            w_loc_loss = tf.reduce_sum(tf.nn.relu(self.min_spill_frac-spill_frac))

            total_loss = cls_loss + self.loc_coeff*w_loc_loss

            '''_,sims_all = self(X, training=True)  # Forward pass
            sims_top_k = tf.math.top_k(sims_all,k=self.top_k,sorted=False).values

            sp = tf.transpose(sims_top_k[:32])
            sn = tf.transpose(sims_top_k[32:])

            total_loss = self.circle_loss(sp, sn, margin=FLAGS.all_margin)'''

            ''''sims_all = tf.matmul(embds[:32],embds,transpose_b=True)
            sims_all = tf.reshape(sims_all,[32,96])
            sp_all = sims_all[:,:32]
            sn_all = sims_all[:,32:]

            sp_puddle = sims_all[:16,:16]
            sn_puddle = sims_all[:16,32:64]
            sp_video = sims_all[16:24,16:24]
            sn_video = sims_all[16:24,64:80]

            loss_all = self.circle_loss(sp_all, sn_all, margin=FLAGS.all_margin)
            loss_puddle = self.circle_loss(sp_puddle, sn_puddle, margin=FLAGS.puddle_margin)
            loss_video = self.circle_loss(sp_video, sn_video, margin=FLAGS.vid_margin)

            total_loss = loss_all + FLAGS.puddle_coeff*loss_puddle + FLAGS.vid_coeff*loss_video'''

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(cls_loss)
        loc_loss_tracker.update_state(w_loc_loss)
        acc_tracker.update_state(y,pred)
        locnet_tracker.update_state(tf.reduce_mean(spill_frac))
        #all_tracker.update_state(loss_all)
        #puddle_tracker.update_state(loss_puddle)
        #video_tracker.update_state(loss_video)
        #return {"loss": loss_tracker.result(), "all_loss": all_tracker.result(), "puddle_loss": puddle_tracker.result(), "video_loss": video_tracker.result()}

        return {"loss": loss_tracker.result(),"acc": acc_tracker.result(), "spill_frac":locnet_tracker.result(), "loc_loss":loc_loss_tracker.result()}

    def test_step(self, data):
        self.zeros = self.zeros20

        X,lab = data
        masks,loc_reg_pts = lab
        y = tf.reduce_max(masks,axis=[1,2])

        pred,theta = self(X,training=False)
        cls_loss = self.compiled_loss(y,pred)

        mask_crops = spatial_transform(masks[:4], theta[:4])
        spill_frac = tf.reduce_mean(mask_crops,axis=[1,2,3])

        t_x = theta[:4,2:3]
        t_y = theta[:4,5:6]
        x_diff = tf.reduce_mean(t_x-loc_reg_pts[:,:,1],axis=1)
        y_diff = tf.reduce_mean(t_y-loc_reg_pts[:,:,0],axis=1)
        loc_loss = x_diff**2 + y_diff**2
        w_loc_loss = tf.reduce_sum(tf.nn.relu(self.min_spill_frac-spill_frac)*loc_loss)

        total_loss = cls_loss + self.loc_coeff*w_loc_loss

        '''_,sims_all = self(X, training=False)  # Forward pass
        sims_top_k = tf.math.top_k(sims_all,k=1,sorted=False).values

        sp = tf.transpose(sims_top_k[:16])
        sn = tf.transpose(sims_top_k[16:])

        total_loss = self.circle_loss(sp, sn, margin=FLAGS.all_margin)'''

        '''sims_all = tf.matmul(embds[:16],embds,transpose_b=True)
        sp_all = sims_all[:,:16]
        sn_all = sims_all[:,16:]
        sp_video = sims_all[:8,:8]
        sn_video = sims_all[:8,16:32]

        loss_all = self.circle_loss(sp_all, sn_all, margin=FLAGS.all_margin)
        loss_puddle = 0.
        loss_video = self.circle_loss(sp_video, sn_video, margin=FLAGS.vid_margin)

        total_loss = loss_all + FLAGS.puddle_coeff*loss_puddle + FLAGS.vid_coeff*loss_video'''

        loss_tracker.update_state(cls_loss)
        loc_loss_tracker.update_state(w_loc_loss)
        acc_tracker.update_state(y,pred)
        locnet_tracker.update_state(tf.reduce_mean(spill_frac))
        #all_tracker.update_state(loss_all)
        #puddle_tracker.update_state(loss_puddle)
        #video_tracker.update_state(loss_video)
        #return {"loss": loss_tracker.result(), "all_loss": all_tracker.result(), "puddle_loss": puddle_tracker.result(), "video_loss": video_tracker.result()}

        return {"loss": loss_tracker.result(),"acc": acc_tracker.result(), "spill_frac":locnet_tracker.result(), "loc_loss":loc_loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        #return [loss_tracker, all_tracker, puddle_tracker, video_tracker]

        return [loss_tracker,acc_tracker,locnet_tracker,loc_loss_tracker]

    def circle_loss(self, sp, sn, margin):
        """ use within-class similarity and between-class similarity for loss
        Args:
            sp (tf.Tensor): within-class similarity  shape [batch, K]
            sn (tf.Tensor): between-class similarity shape [batch, L]
        Returns:
            tf.Tensor: loss
        """
        Delta_p = 1 - margin
        Delta_n = margin

        ap = tf.nn.relu(-tf.stop_gradient(sp) + 1 + margin)
        an = tf.nn.relu(tf.stop_gradient(sn) + margin)

        logit_p = -ap * (sp - Delta_p) * self.gamma
        logit_n = an * (sn - Delta_n) * self.gamma

        return tf.reduce_mean(tf.math.softplus(
            tf.math.reduce_logsumexp(logit_n, axis=-1, keepdims=True) +
            tf.math.reduce_logsumexp(logit_p, axis=-1, keepdims=True)))