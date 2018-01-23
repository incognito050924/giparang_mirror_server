import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.framework import arg_scope

def weight_initializer(func_name='trunc_norm', stddev=0.01):
    """
    Weight Tensors를 초기화하는데 사용되는 함수를 리턴한다.
    :param func_name: ['trunc_norm'(default), 'xavier', 'variance_scaling']
    :param stddev: The float number what used to truncated normal initializer
    :return: initialize function object
    """
    initializer = None
    if func_name == 'trunc_norm':
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=stddev)
    elif func_name == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.contrib.layers.variance_scaling_initializer()

    return initializer

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])]

    return kernel_size_out


# Convolution Neural Networks
def simple_cnn(inputs, num_classes, is_training=True):
    pass

def mini_xception():
    pass

def inception_v3(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.8, min_depth=16,
                 depth_multiplier=1.0, prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None,
                 scope='InceptionV3'):
    """
    Inception version 3 model
    :param inputs:
    :param num_classes:
    :param is_training:
    :param dropout_keep_prob:
    :param min_depth:
    :param depth_multiplier:
    :param prediction_fn:
    :param spatial_squeeze:
    :param reuse:
    :param scope:
    :return:
    """

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v3_base(inputs, num_classes, scope=scope, min_depth=min_depth,
                                                depth_multiplier=depth_multiplier)

            # Auxiliary Head logits
            with arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID',
                                                 scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1], scope='Conv2d_1b_1x1')

                    # Shape of feature map before the final layer.
                    kernel_size = _reduced_kernel_size_for_small_input(aux_logits, [5, 5])
                    aux_logits = slim.conv2d(aux_logits, depth(768), kernel_size, padding='VALID',
                                             weights_initializer=weight_initializer('trunc_norm', 0.01),
                                             scope='Conv2d_2a_{}x{}'.format(*kernel_size))
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None,
                                             normalizer_fn=None,
                                             weights_initializer=weight_initializer('trunc_norm', 0.001),
                                             scope='Conv2d_2b_1x1')
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits

            # Final pooling and prediction
            with tf.variable_scope('Logits'):
                kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
                net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                      scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # 1*1*2048
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                # 2048
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                     scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            # 1000
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points

def inception_v3_base(inputs, num_classes, depth_multiplier=1.0, min_depth=16, final_endpoint='Mixed_7c',
                      is_training=True, scope=None):
    """
    Layers            | Scopes
    =======================================
    conv0             | Conv2d_1a_3x3
    conv1             | Conv2d_2a_3x3
    conv2             | Conv2d_2b_3x3
    pool1             | MaxPool_3a_3x3
    conv3             | Conv2d_3b_1x1
    conv4             | Conv2d_4a_3x3
    pool2             | MaxPool_5a_3x3
    mixed_35x35x256   | Mixed_5b
    mixed_35x35x288   | Mixed_5c
    mixed_35x35x288   | Mixed_5d
    mixed_17x17x768   | Mixed_6a
    mixed_17x17x768   | Mixed_6b
    mixed_17x17x768   | Mixed_6c
    mixed_17x17x768   | Mixed_6d
    mixed_17x17x768   | Mixed_6e
    mixed_8x8x1280    | Mixed_7a
    mixed_8x8x2048    | Mixed_7b
    mixed_8x8x2048    | Mixed_7c
    """
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    batch_norm_params = {'is_training': is_training, 'decay': 0.9
        , 'updates_collections': None}
    with tf.variable_scope(scope, 'Inception_V3', [inputs]):
        with arg_scope(
                [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                stride=1, padding='VALID'):

            # input size = 299*299*3
            end_point = 'Conv2d_1a_3x3'
            net = slim.conv2d(inputs=inputs, num_outputs=depth(32), kernel_size=[3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # input size = 149*149*32
            end_point = 'Conv2d_2a_3x3'
            net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)  # stride=1 as default.
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # input size = 147*147*32
            end_point = 'Conv2d_2b_3x3'
            net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # input size = 147*147*64
            end_point = 'MaxPool_3a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # input size = 73*73*64
            end_point = 'Conv2d_3b_1x1'
            net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # input size = 73*73*80.
            end_point = 'Conv2d_4a_3x3'
            net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

            # input size = 71*71*192.
            end_point = 'MaxPool_5a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net, end_points

    # Inception block
    with arg_scope(
            [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
            stride=1, padding='SAME'):

        # input size = 35*35*192
        # mixed: 35*35*256
        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(64), [1, 1], padding='SAME', scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(48), [1, 1], padding='SAME', scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], padding='SAME', scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(64), [1, 1], padding='SAME', scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], padding='SAME', scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], padding='SAME', scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], padding='SAME', scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(32), [1, 1], padding='SAME', scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 35*35*256
        # mixed_1: 35*35*288
        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 35*35*288
        # mixed_2: 35*35*288
        end_point = 'Mixed_5d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 35*35*288
        # mixed_3: 17*17*768
        end_point = 'Mixed_6a'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2, padding='VALID',
                                       scope='Conv2d_1a_1x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 17*17*768
        # mixed_4: 17*17*768
        end_point = 'Mixed_6b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 17*17*768
        # mixed_5: 17*17*768
        end_point = 'Mixed_6c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 17*17*768
        # mixed_6: 17*17*768
        end_point = 'Mixed_6d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 17*17*768
        # mixed_7: 17*17*768
        end_point = 'Mixed_6e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 17*17*768
        # mixed_8: 8*8*1280.
        end_point = 'Mixed_7a'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2, padding='VALID',
                                       scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2, padding='VALID',
                                       scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 8*8*1280
        # mixed_9: 8*8*2048
        end_point = 'Mixed_7b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat(
                    [
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')
                    ], 3)
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = tf.concat(
                    [
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                    ], 3)
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points

        # input size = 8*8*2048
        # mixed_10: 8*8*2048
        end_point = 'Mixed_7c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat(
                    [
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')
                    ], 3)
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = tf.concat(
                    [
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')
                    ], 3)
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        end_points[end_point] = net
        if end_point == final_endpoint:
            return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)
