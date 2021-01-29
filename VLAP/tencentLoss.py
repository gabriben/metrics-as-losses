from .hyperparameters import *
import tensorflow as tf


# TENCENT (original descriptions to the right)

TENCENT_MASK_THRESH = 0.7 # "mask thres for balance pos neg"
TENCENT_CLASS_NUM = 1000 # "distinct class number"
TENCENT_RANDOM_SEED = 1234 #  "Random sedd for neigitive class selected"
TENCENT_WEIGHT_DECAY = 0.0001 # "Tainable Weight l2 loss factor."
TENCENT_NEG_SELECT = 0.3 # "how many class within only negtive samples in a batch select to learn"



def tencentLoss(labels, logits):
# Calculate loss, which includes softmax cross entropy and L2 regularization.
  # a. get loss coeficiente
  pos_mask = tf.reduce_sum(
               tf.cast(
                 tf.math.greater_equal(
                   labels, tf.fill(tf.shape(labels), TENCENT_MASK_THRESH)), 
                   tf.float32), 
             0)
  pos_curr_count = tf.cast(tf.math.greater(pos_mask, 0), tf.float32)
  neg_curr_count = tf.cast(tf.less_equal(pos_mask, 0), tf.float32)
  pos_count = tf.Variable(tf.zeros(shape=[TENCENT_CLASS_NUM,]),  trainable=False)
  neg_count = tf.Variable(tf.zeros(shape=[TENCENT_CLASS_NUM,]),  trainable=False)
  neg_select = tf.cast(
                 tf.less_equal(
                    tf.random.uniform( # random_uniform
                      shape=[TENCENT_CLASS_NUM,], 
                      minval=0, maxval=1,
                      seed = TENCENT_RANDOM_SEED),
                    TENCENT_NEG_SELECT), 
                 tf.float32)
  tf.summary.histogram('pos_curr_count', pos_curr_count)
  tf.summary.histogram('neg_curr_count', neg_curr_count)
  tf.summary.histogram('neg_select', neg_select)
  tf.print(pos_count)
  tf.print(pos_curr_count)  
  tf.print(neg_curr_count)  
  with tf.control_dependencies([pos_curr_count, neg_curr_count, neg_select]):
    pos_count = tf.compat.v1.assign_sub( # modif ici: + v1
                   tf.compat.v1.assign_add(pos_count, pos_curr_count),
                   tf.math.multiply(pos_count, neg_curr_count))
    neg_count = tf.compat.v1.assign_sub(
                   tf.compat.v1.assign_add(neg_count, tf.math.multiply(neg_curr_count, neg_select)),
                   tf.math.multiply(neg_count, pos_curr_count))
    tf.summary.histogram('pos_count', pos_count)
    tf.summary.histogram('neg_count', neg_count)
  pos_loss_coef = -1 * (tf.log((0.01 + pos_count)/10)/tf.log(10.0))
  pos_loss_coef = tf.where(
                    tf.math.greater(pos_loss_coef, tf.fill(tf.shape(pos_loss_coef), 0.01)),
                    pos_loss_coef,
                    tf.fill(tf.shape(pos_loss_coef), 0.01))
  pos_loss_coef = tf.math.multiply(pos_loss_coef, pos_curr_count)
  tf.summary.histogram('pos_loss_coef', pos_loss_coef)
  neg_loss_coef = -1 * (tf.log((8 + neg_count)/10)/tf.log(10.0))
  neg_loss_coef = tf.where(
                   tf.math.greater(neg_loss_coef, tf.fill(tf.shape(neg_loss_coef), 0.01)),
                   neg_loss_coef,
                   tf.fill(tf.shape(neg_loss_coef), 0.001))
  neg_loss_coef = tf.math.multiply(neg_loss_coef, tf.math.multiply(neg_curr_count, neg_select))
  tf.summary.histogram('neg_loss_coef', neg_loss_coef)
  loss_coef = tf.math.add(pos_loss_coef, neg_loss_coef)
  tf.summary.histogram('loss_coef', loss_coef)

  # b. get non-negative mask
  non_neg_mask = tf.fill(tf.shape(labels), -1.0, name='non_neg')
  non_neg_mask = tf.cast(tf.not_equal(labels, non_neg_mask), tf.float32)
  tf.summary.histogram('non_neg', non_neg_mask)

  # cal loss
  cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
       logits=logits, targets=labels, pos_weight=12, name='sigmod_cross_entropy')
  tf.summary.histogram('sigmod_ce', cross_entropy)
  cross_entropy_cost = tf.reduce_sum(tf.reduce_mean(cross_entropy * non_neg_mask, axis=0) * loss_coef)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy_cost, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy_cost)

  # Add weight decay to the loss. We exclude the batch norm variables because
  # doing so leads to a small improvement in accuracy.
  loss = cross_entropy_cost + TENCENT_WEIGHT_DECAY * tf.math.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name])
  return loss
