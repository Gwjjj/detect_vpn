
import tensorflow as tf
import math
l = tf.keras.layers
se_ratio = 0.25
weight_decay = 1e-5
def _conv(inputs, filters, kernel_size, strides, bias=False, normalize=True, activation='swish'):
    output = inputs
    padding_str = 'same'
    output = l.Conv2D(
        filters, kernel_size, strides, padding_str, use_bias=bias, \
        kernel_initializer='he_normal', data_format='channels_first', \
        kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)
    )(output)
    if normalize:
        output = l.BatchNormalization(axis=1)(output)
    if activation=='relu':
        output = l.ReLU()(output)
    if activation=='relu6':
        output = l.ReLU(max_value=6)(output)
    if activation=='leaky_relu':
        output = l.LeakyReLU(alpha=0.1)(output)
    if activation=='sigmoid':
        output = l.Activation('sigmoid')(output)
    if activation=='swish':
        output = l.Activation(tf.nn.swish)(output)
    return output
 
def _dwconv(inputs, kernel_size, strides, bias=True, activation='swish'):
    output = inputs
    padding_str = 'same'
    output = l.DepthwiseConv2D(
        kernel_size, strides, padding_str, use_bias=bias, data_format='channels_first', \
        depthwise_initializer='he_uniform', depthwise_regularizer=tf.keras.regularizers.l2(l=weight_decay))(output)
    output = l.BatchNormalization(axis=1)(output)
    if activation=='relu':
        output = l.ReLU()(output)
    if activation=='relu6':
        output = l.ReLU(max_value=6)(output)
    if activation=='leaky_relu':
        output = l.LeakyReLU(alpha=0.1)(output)
    if activation=='swish':
        output = l.Activation(tf.nn.swish)(output)
    return output
 
def _bottleneck(inputs, in_filters, out_filters, kernel_size, strides, bias=False, activation='swish', t=1):
    output = _conv(inputs, in_filters*t, 1, 1, False, True, activation)
    output = _dwconv(output, kernel_size, strides, False, activation)
    #SE Layer
    se_output = tf.reduce_mean(output, axis=[2,3], keepdims=True)
    squeeze_filter = max(1, int(in_filters*se_ratio))
    se_output = _conv(se_output, squeeze_filter, 1, 1, True, False, activation)
    se_output = _conv(se_output, in_filters*t, 1, 1, True, False, 'sigmoid')
    output = se_output*output
    #Conv
    output = _conv(output, out_filters, 1, 1, False, True, 'linear')
    if strides==1 and in_filters==out_filters:
        output = l.add([output, inputs])
    return output
 
def _block(inputs, in_filters, out_filters, kernel_size, strides, bias=False, activation='swish', t=1, repeats=1):
    output = _bottleneck(inputs, in_filters, out_filters, kernel_size, strides, bias, activation, t)
    for i in range(repeats-1):
        output = _bottleneck(output, out_filters, out_filters, kernel_size, 1, bias, activation, t)
    return output
 
def round_filters(filters, beta):
    divisor = 8
    filters *= beta
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)
 
def round_repeats(repeats, alpha):
    return int(math.ceil(alpha * repeats))
 
def efficientnet_model(alpha,beta,gamma,dropout):
    # Input Layer
    image = tf.keras.Input(shape=(3,None,None))                        #224*224*3
    out_filter = round_filters(32, beta)
    net = _conv(image, out_filter, 3, 2, False, True, 'swish')            #112*112*32
    #MBConv Block 1
    repeats = round_repeats(1, alpha)
    in_filter = round_filters(32, beta)
    out_filter = round_filters(16, beta)
    net = _block(net, in_filter, out_filter, 3, 1, False, 'swish', 1, repeats)    #112*112*16
    #MBConv Block 2
    repeats = round_repeats(2, alpha)
    in_filter = round_filters(16, beta)
    out_filter = round_filters(24, beta)
    net = _block(net, in_filter, out_filter, 3, 2, False, 'swish', 6, repeats)    #56*56*24
    #MBConv Block 3
    repeats = round_repeats(2, alpha)
    in_filter = round_filters(24, beta)
    out_filter = round_filters(40, beta)
    net = _block(net, in_filter, out_filter, 5, 2, False, 'swish', 6, repeats)    #28*28*40
    #MBConv Block 4
    repeats = round_repeats(3, alpha)
    in_filter = round_filters(40, beta)
    out_filter = round_filters(80, beta)
    net = _block(net, in_filter, out_filter, 3, 2, False, 'swish', 6, repeats)    #14*14*80
    #MBConv Block 5
    repeats = round_repeats(3, alpha)
    in_filter = round_filters(80, beta)
    out_filter = round_filters(112, beta)
    net = _block(net, in_filter, out_filter, 5, 1, False, 'swish', 6, repeats)    #14*14*112
    #MBConv Block 6
    repeats = round_repeats(4, alpha)
    in_filter = round_filters(112, beta)
    out_filter = round_filters(192, beta)
    net = _block(net, in_filter, out_filter, 5, 2, False, 'swish', 6, repeats)    #7*7*192
    #MBConv Block 7
    repeats = round_repeats(1, alpha)
    in_filter = round_filters(192, beta)
    out_filter = round_filters(320, beta)
    net = _block(net, in_filter, out_filter, 3, 1, False, 'swish', 6, repeats)    #7*7*320
    #Conv
    out_filter = round_filters(1280, beta)
    net = _conv(net, out_filter, 1, 1, False, True, 'swish')            #7*7*1280
    net = tf.reduce_mean(net, axis=[2,3], keepdims=False)                 #GlobalPool, 1280
    net = l.Dropout(rate=dropout)(net)
    logits = l.Dense(units=1000, name='output')(net)
    model = tf.keras.Model(inputs=image, outputs=logits)
    return model
