import numpy as np
import tensorflow as tf


def tensorflow_convolution_full(x_tens,y_tens, DTYPE=tf.float64):
    """
    expects two 1D tensors x_tens and y_tens with

    shape x_tens = (N)
    shape y_tens = (M)

    returns tensor result with

    shape result = (M + N -1)

    equivalently like numpy.convolve(x,y,mode="full") does. For efficiency, x_tens and y_tens might get swapped.
    """

    # swap if y is longer than x
    if x_tens.get_shape() > y_tens.get_shape():
        a_tens = x_tens
        b_tens = y_tens
    else:
        a_tens = y_tens
        b_tens = x_tens

    M = int(b_tens.get_shape()[0])

    # calculate padding for a_tens
    if M % 2 == 1:
        left_padding = (M-1)//2
        right_padding = (M-1)//2
    else:
        left_padding = M // 2
        right_padding = (M-2) // 2

    # pad a_tens
    a_tens_padded = tf.concat((tf.zeros(left_padding, dtype=DTYPE),a_tens, tf.zeros(right_padding, dtype=DTYPE)), axis=0)
    # reverse b_tens
    b_tens_rev = tf.reverse(b_tens, axis=[0,])

    a_tens_padded_4D = tf.reshape(a_tens_padded, (1, a_tens_padded.get_shape()[0], 1, 1))
    b_tens_rev_4D = tf.reshape(b_tens_rev, (b_tens_rev.get_shape()[0], 1, 1, 1))
    tens_conved = tf.nn.conv2d(a_tens_padded_4D, b_tens_rev_4D, (1,1,1,1), "SAME")

    return tf.reshape(tens_conved, (tens_conved.get_shape()[1],))


def tensorflow_convolution_full_4D_inputs(x_tens,y_tens, DTYPE=tf.float64):
    """
    expects two 1D tensors x_tens and y_tens with

    shape x_tens = (1,N,1,1)
    shape y_tens = (M,1,1,1)

    returns tensor result with

    shape result = (M+N-1, 1, 1, 1)

    similarly like numpy.convolve(x,y,mode="full") does. For efficiency, x_tens and y_tens might get swapped.
    """

    # swap if y is longer than x
    if x_tens.get_shape() > y_tens.get_shape():
        a_tens = x_tens
        b_tens = y_tens
    else:
        a_tens = tf.reshape(y_tens, (1,y_tens.get_shape()[2],1,1))
        b_tens = tf.reshape(x_tens, (x_tens.get_shape()[1],1,1,1))

    M = int(b_tens.get_shape()[0])

    # calculate padding for a_tens
    if M % 2 == 1:
        left_padding = (M-1)//2
        right_padding = (M-1)//2
    else:
        left_padding = M // 2
        right_padding = (M-2) // 2

    # pad a_tens
    a_tens_padded = tf.concat((tf.zeros((1,left_padding,1,1), dtype=DTYPE), a_tens, tf.zeros((1,right_padding,1,1), dtype=DTYPE)), axis=1)
    # reverse b_tens
    b_tens_rev = tf.reverse(b_tens, axis=[0])

    return tf.nn.conv2d(a_tens_padded, b_tens_rev, (1,1,1,1), "SAME")


if __name__ == "__main__":

    print("1D inputs")

    x = np.arange(4)
    y = np.arange(5,10)

    print(np.convolve(x,y, mode="full"))

    x_tens = tf.placeholder(dtype=tf.float64, shape=(len(x)))
    y_tens = tf.placeholder(dtype=tf.float64, shape=(len(y)))

    tens_conv = tensorflow_convolution_full(x_tens, y_tens)

    with tf.Session() as sess:
        print(sess.run(tens_conv, feed_dict={x_tens: x, y_tens: y}))

    print("4D inputs")

    x = np.arange(4)
    y = np.arange(5,10)
    x_4D = np.reshape(x, (1,len(x),1,1))
    y_4D = np.reshape(y, (len(y),1,1,1))

    print(np.convolve(x,y, mode="full"))

    x_tens = tf.placeholder(dtype=tf.float64, shape=x_4D.shape)
    y_tens = tf.placeholder(dtype=tf.float64, shape=y_4D.shape)

    tens_conv = tensorflow_convolution_full_4D_inputs(x_tens, y_tens)

    with tf.Session() as sess:
        res = sess.run(tens_conv, feed_dict={x_tens: x_4D, y_tens: y_4D})
        print(res[0,:,0,0])
