import os

batch_size = 32
eval_dir = str(os.path.join(os.getcwd(), 'api', 'ml', 'trained', 'fer2013_eval'))
checkpoint_dir = str(os.path.join(os.getcwd(), 'api', 'ml', 'trained', 'train_dir'))
eval_interval_secs = 60 * 5
run_once = False
num_examples = 7178
subset = 'validation'
image_size = 299
image_depth = 1
num_preprocess_threads = 1
num_readers = 4
input_queue_memory_factor = 16
