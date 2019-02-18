import numpy as np
import scipy
import tensorflow as tf

"""Implement simple version of ELG architecture as introduced in
https://github.com/swook/GazeML [Park et al. ETRA'18]."""

# model parameters        
data_format == 'NHWC' 
data_format_longer = ('channels_first' if data_format == 'NCHW'
                                        else 'channels_last')
hg_num_feature_maps = 32
hg_first_layer_stride = 1
num_modules = 2
hg_num_residual_blocks = 1
hg_num_landmarks = 18

def inference(input_eye, placeholder_output_size):
    x = input_eye
   
    outputs = {}

    with tf.variable_scope('hourglass'):
        hg_num_landmarks == 18

        # Prepare for Hourglass by downscaling via conv
        with tf.variable_scope('pre'):
            n = hg_num_feature_maps
            x = apply_conv(x, num_features=n, kernel_size=7, stride=hg_first_layer_stride)
            x = tf.nn.relu(apply_bn(x))
            x = build_residual_block(x, n, 2*n, name='res1')
            x = build_residual_block(x, 2*n, n, name='res2')

        # Hourglass blocks
        x_prev = x
        for i in range(hg_num_modules):
            with tf.variable_scope('hg_%d' % (i + 1)):
                x = build_hourglass(x, steps_to_go=4, num_features=hg_num_feature_maps)
                merge_flag = (i < (hg_num_modules - 1))
                x, h = build_hourglass_after(x_prev, x, do_merge=merge_flag))
                x_prev = x
        x = h

        outputs['heatmaps'] = x

    # Soft-argmax
    x = calculate_landmarks(x)
    with tf.variable_scope('upscale'):
        # Upscale since heatmaps are half-scale of original image
        x *= hg_first_layer_stride
        outputs['landmarks'] = x

    # Fully-connected layers for radius regression
    with tf.variable_scope('radius'):
        x = tf.contrib.layers.flatten(tf.transpose(x, perm=[0, 2, 1]))
        for i in range(3):
            with tf.variable_scope('fc%d' % (i + 1)):
                x = tf.nn.relu(apply_bn(apply_fc(x, 100)))
        with tf.variable_scope('out'):
            x = apply_fc(x, 1)
        outputs['radius'] = x

    # Define outputs
    return outputs

def _apply_conv(tensor, num_features, kernel_size=3, stride=1):
    return tf.layers.conv2d(
        tensor,
        num_features,
        kernel_size = kernel_size,
        strides = stride,
        padding = 'SAME',
        kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-4),
        bias_initializer = tf.zeros_initializer(),
        data_format = self._data_format_longer,
        name = 'conv',
    )

def apply_fc(tensor, num_outputs):
    return tf.layers.dense(
        tensor,
        num_outputs,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
        bias_initializer=tf.zeros_initializer(),
        name='fc',
    )

def apply_pool(tensor, kernel_size=3, stride=2):
    tensor = tf.layers.max_pooling2d(
        tensor,
        pool_size = kernel_size,
        strides = stride,
        padding = 'SAME',
        data_format = data_format_longer,
        name='pool',
    )
    return tensor

def apply_bn(tensor):
    #TODO: replace with tf.layers.batch_normalization()
    return tf.contrib.layers.batch_norm(
        tensor,
        scale = True,
        center = True,
        is_training = True,
        trainable = True,
        data_format = data_format,
        updates_collections = None,
    )

def build_residual_block(x, num_in, num_out, name='res_block'):
    with tf.variable_scope(name):
        half_num_out = max(int(num_out/2), 1)
        c = x
        with tf.variable_scope('conv1'):
            c = tf.nn.relu(apply_bn(c))
            c = apply_conv(c, num_features=half_num_out, kernel_size=1, stride=1)
        with tf.variable_scope('conv2'):
            c = tf.nn.relu(apply_bn(c))
            c = apply_conv(c, num_features=half_num_out, kernel_size=3, stride=1)
        with tf.variable_scope('conv3'):
            c = tf.nn.relu(apply_bn(c))
            c = apply_conv(c, num_features=num_out, kernel_size=1, stride=1)
        with tf.variable_scope('skip'):
            if num_in == num_out:
                s = tf.identity(x)
            else:
                s = apply_conv(x, num_features=num_out, kernel_size=1, stride=1)
        x = c + s
    return x

def build_hourglass(x, steps_to_go, num_features, depth=1):
    with tf.variable_scope('depth%d' % depth):
        # Upper branch
        up1 = x
        for i in range(hg_num_residual_blocks):
            up1 = build_residual_block(up1, num_features, num_features,
                                                name='up1_%d' % (i + 1))
        # Lower branch
        low1 = apply_pool(x, kernel_size=2, stride=2)
        for i in range(hg_num_residual_blocks):
            low1 = build_residual_block(low1, num_features, num_features,
                                                name='low1_%d' % (i + 1))
        # Recursive
        low2 = None
        if steps_to_go > 1:
            low2 = build_hourglass(low1, steps_to_go - 1, num_features, depth=depth+1)
        else:
            low2 = low1
            for i in range(hg_num_residual_blocks):
                low2 = build_residual_block(low2, num_features, num_features,
                                                    name='low2_%d' % (i + 1))
        # Additional residual blocks
        low3 = low2
        for i in range(hg_num_residual_blocks):
            low3 = build_residual_block(low3, num_features, num_features,
                                                name='low3_%d' % (i + 1))
        # Upsample
        if data_format == 'NCHW':  # convert to NHWC
            low3 = tf.transpose(low3, (0, 2, 3, 1))
        up2 = tf.image.resize_bilinear(
                low3,
                up1.shape[1:3] if self._data_format == 'NHWC' else up1.shape[2:4],
                align_corners=True,
                )
        if data_format == 'NCHW':  # convert back from NHWC
            up2 = tf.transpose(up2, (0, 3, 1, 2))

    return up1 + up2

def build_hourglass_after(x_prev, x_now, do_merge=True):
    n = hg_num_feature_maps
    with tf.variable_scope('after'):
        for j in range(hg_num_residual_blocks):
            x_now = build_residual_block(x_now, n, n, name='after_hg_%d' % (j + 1))
        x_now = apply_conv(x_now, n, kernel_size=1, stride=1)
        x_now = apply_bn(x_now)
        x_now = tf.nn.relu(x_now)

        with tf.variable_scope('hmap'):
            h = apply_conv(x_now, hg_num_landmarks, kernel_size=1, stride=1)

    x_next = x_now
    if do_merge:
        with tf.variable_scope('merge'):
            with tf.variable_scope('h'):
                x_hmaps = apply_conv(h, n, kernel_size=1, stride=1)
            with tf.variable_scope('x'):
                x_now = apply_conv(x_now, n, kernel_size=1, stride=1)
            x_next += x_prev + x_hmaps
    return x_next, h
