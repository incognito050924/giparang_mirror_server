import os
from .models.cnn import inception_v3
from .utils.datasets import DatasetManager
from .utils.datasets import split_data
import tensorflow as tf

# Hyper Parameters
batch_size = 32
num_epochs = 10000
input_shape = (299, 299, 3) # input_shape = (64, 64, 1)
num_classes = 7
validation_split = .2
val_acc = 0
val_acc_max = 0
is_training = tf.placeholder(tf.bool)

# Load Data
data_loader = DatasetManager('fer2013')
x_data, y_data = data_loader.get_data()
num_samples, num_classes = y_data.shape
train_data, val_data = split_data(x_data, y_data, validation_split)
train_faces, train_emotions = train_data
val_faces,  val_emotions = val_data

# Define Graph
# Placeholder
x = tf.placeholder("float", [None, 299, 299, 3])
y = tf.placeholder("float", [None, num_classes])

# Prediction
prediction, end_points = inception_v3(x, num_classes=num_classes, is_training=True)
# Loss & Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializer
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Confirm Trainable Variables
# trainable_weights = tf.trainable_variables()
# for i, weight_name in enumerate(trainable_weights):
#     wval = sess.run(weight_name)
#     print("[%d/%d]\tShape: %s\t[%s]" % (i, len(trainable_weights), wval.shape, weight_name))

# Saver
savedir = 'nets/slim_inception_v3/'
saver = tf.train.Saver(max_to_keep=100)
save_step = 4
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Train model(Optimize)
for epoch in range(num_epochs):
    avg_cost = 0.