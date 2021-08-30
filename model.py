import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

from absl import flags, app

FLAGS = flags.FLAGS

loss_tracker = tf.keras.metrics.Mean(name="loss")
all_tracker = tf.keras.metrics.Mean(name="loss")
puddle_tracker = tf.keras.metrics.Mean(name="loss")
video_tracker = tf.keras.metrics.Mean(name="loss")

class PrototypeSim(tf.keras.layers.Layer):
    def __init__(self):
        super(PrototypeSim, self).__init__()
        self.prototypes = tf.Variable(tf.random.normal((320,FLAGS.num_prototypes)))

    def call(self, inputs):
        embds = tf.nn.l2_normalize(inputs, axis=-1)
        prototypes = tf.nn.l2_normalize(self.prototypes, axis=-1)
        sims = tf.matmul(embds,prototypes)

        return sims

class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)

        self.conv_base = EfficientNetB0(weights="effnet_weights/efficientnetb0_notop.h5", include_top=False, input_shape=(224,224,3))
        self.max_pool = layers.GlobalMaxPooling2D(name="gap")
        self.proj_head = layers.Dense(320, activation="linear", name="fc_out")
        self.prototype_layer = PrototypeSim()
        
        self.gamma = FLAGS.gamma
        self.top_k = FLAGS.top_k

    def call(self, inputs, training=False):
        x = self.conv_base(inputs, training=training)
        x = self.max_pool(x)
        x = self.proj_head(x)

        sims = self.prototype_layer(x)

        return x,sims

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        X,y = data

        with tf.GradientTape() as tape:

            _,sims_all = self(X, training=True)  # Forward pass
            sims_top_k = tf.math.top_k(sims_all,k=self.top_k,sorted=False).values

            sp = tf.transpose(sims_top_k[:32])
            sn = tf.transpose(sims_top_k[32:])

            total_loss = self.circle_loss(sp, sn, margin=FLAGS.all_margin)

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

        loss_tracker.update_state(total_loss)
        #all_tracker.update_state(loss_all)
        #puddle_tracker.update_state(loss_puddle)
        #video_tracker.update_state(loss_video)
        #return {"loss": loss_tracker.result(), "all_loss": all_tracker.result(), "puddle_loss": puddle_tracker.result(), "video_loss": video_tracker.result()}

        return {"loss": loss_tracker.result()}

    def test_step(self, data):

        X,y = data

        _,sims_all = self(X, training=False)  # Forward pass
        sims_top_k = tf.math.top_k(sims_all,k=self.top_k,sorted=False).values

        sp = tf.transpose(sims_top_k[:32])
        sn = tf.transpose(sims_top_k[32:])

        total_loss = self.circle_loss(sp, sn, margin=FLAGS.all_margin)

        '''sims_all = tf.matmul(embds[:16],embds,transpose_b=True)
        sp_all = sims_all[:,:16]
        sn_all = sims_all[:,16:]
        sp_video = sims_all[:8,:8]
        sn_video = sims_all[:8,16:32]

        loss_all = self.circle_loss(sp_all, sn_all, margin=FLAGS.all_margin)
        loss_puddle = 0.
        loss_video = self.circle_loss(sp_video, sn_video, margin=FLAGS.vid_margin)

        total_loss = loss_all + FLAGS.puddle_coeff*loss_puddle + FLAGS.vid_coeff*loss_video'''

        loss_tracker.update_state(total_loss)
        #all_tracker.update_state(loss_all)
        #puddle_tracker.update_state(loss_puddle)
        #video_tracker.update_state(loss_video)
        #return {"loss": loss_tracker.result(), "all_loss": all_tracker.result(), "puddle_loss": puddle_tracker.result(), "video_loss": video_tracker.result()}

        return {"loss": loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        #return [loss_tracker, all_tracker, puddle_tracker, video_tracker]

        return [loss_tracker]

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