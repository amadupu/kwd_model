
from data_decoder import TFDecoder
import tensorflow as tf
import numpy as np
if __name__ == '__main__':
    decoder = TFDecoder.Builder(). \
        set_feature_size(26). \
        set_num_epochs(1). \
        set_path('records/eval'). \
        set_shuffle_status(False). \
        build()

    batch_input = tf.train.batch(tensors=decoder.dequeue(), batch_size=1,
                                                  dynamic_pad=True,
                                                  allow_smaller_final_batch=True)

    # def func1():
    #     delta = 50 - tf.shape(batch_input[1])[1]
    #     paddings = tf.pad(batch_input[1],tf.constant([[0,0],[0,delta],[0,0]]),'CONSTANT')
    #     return paddings
    #
    # def func2():
    #     return batch_input[1]
    #
    # r = tf.cond(tf.less(tf.cast(tf.shape(batch_input[1])[1],tf.float32),tf.constant(50,tf.float32)),func1, func2)

    l_init = tf.global_variables_initializer()
    g_init = tf.local_variables_initializer()

    # y_reshaped = tf.reshape(batch_input[2],[-1])

    with tf.Session() as sess:
        sess.run([l_init,g_init])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            while not coord.should_stop():
                result = sess.run(batch_input)

                # print(result[0], result[1], result[2], result[3])
                # print(np.shape(result[4]),np.shape(result[5]))
                print(result[0],result[1],result[2],result[3],np.shape(result[4]), np.shape(result[5]))
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()

        coord.join(threads=threads)

