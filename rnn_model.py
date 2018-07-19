import tensorflow as tf
from data_decoder import TFDecoder
from enum import Enum
import numpy as np
import features
import os

class RNNModel(object):
    def __init__(self,builder):

        self.epochs = builder.epochs
        # batch size
        self.batch_size = builder.batch_size
        self.read_path = builder.read_path
        self.feature_size = builder.feature_size
        self.num_layers = builder.num_layers
        self.max_steps = builder.max_steps
        self.num_classes = builder.num_classes
        self.cell_size = builder.cell_size
        self.cell_type = builder.cell_type
        self.learning_rate = builder.learning_rate
        self.threads= None
        self.batch_inputs = None
        self.evalfn = builder.evalfn
        self.oper_mode = builder.oper_mode
        self.global_step = None
        self.validation_step = builder.validation_step
        self.logs_path = builder.logs_path
        self.model_path = builder.model_path
        self.model_name = builder.model_name
        self.is_classifier = builder.is_classifier
        self.build_graph()

    def build_graph(self):
        with tf.Graph().as_default():

            # if self.oper_mode == RNNModel.OperMode.TEST or \
            #         self.oper_mode == RNNModel.OperMode.OPER_MODE_EVAL:
            #     filename = ".".join([tf.train.latest_checkpoint(self.model_path),'meta'])
            #     self.saver = tf.train.import_meta_graph(filename)
            # else:
                with tf.name_scope('input_pipe_line'):
                    if self.oper_mode == RNNModel.OperMode.OPER_MODE_TEST:
                        self.xs = tf.placeholder(tf.float32,[None,self.max_steps,self.feature_size],name='xs')
                        # pad's second argument can be seen as [[up, down], [left, right]]
                        if self.is_classifier:
                            self.ys = tf.placeholder(tf.int64, [None], name='ys')
                        else:
                            self.ys = tf.placeholder(tf.int64, [None, self.max_steps], name='ys')

                        self.steps = tf.placeholder(tf.int64, [None], name='steps')
                    else:
                        decoder = TFDecoder.Builder(). \
                            set_feature_size(self.feature_size). \
                            set_num_epochs(self.epochs). \
                            set_path(self.read_path). \
                            set_shuffle_status(True). \
                            build()
                        self.steps, self.xs, self.ys = tf.train.batch(tensors=decoder.dequeue(self.is_classifier), batch_size=self.batch_size,
                                                                  dynamic_pad=True,
                                                                  allow_smaller_final_batch=True,name='batch_processor')
                        self.global_step = tf.Variable(0, name="global_step", trainable=False)

                self.keepprob = tf.placeholder(tf.float32, [], name='keeprob')

                # input weights
                with tf.name_scope('input_layer'):
                    with tf.name_scope('Weigths'):
                        Win = self.weight_variable([self.feature_size,self.cell_size],name='W_in')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('input_layer/Weights',Win)
                    with tf.name_scope('Biases'):
                        Bin = self.bias_variable([self.cell_size], name='B_in')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('input_layer/Biases', Bin)
                    xs = tf.reshape(self.xs,[-1,self.feature_size])
                    rnn_inputs = tf.add(tf.matmul(xs,Win),Bin)
                    rnn_inputs = tf.nn.dropout(rnn_inputs,keep_prob=self.keepprob)
                    self.rnn_inputs = tf.reshape(rnn_inputs,[-1,self.max_steps,self.cell_size],name='rnn_inputs')

                with tf.name_scope('rnn_layer'):
                    if self.cell_type == RNNModel.CellType.RNN_CEL_TYPE_LSTM:
                        cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, state_is_tuple=True,name='LSTMCell')
                    else:
                        cell = tf.nn.rnn_cell.GRUCell(self.cell_size,name='GRUCell')

                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=self.keepprob)

                    if self.cell_type == RNNModel.CellType.RNN_CEL_TYPE_LSTM:
                        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)
                    else:
                        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)

                    cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=self.keepprob)

                    batch_size = tf.shape(self.xs)[0]

                    self.tf_batch_size = batch_size

                    self.state = cell.zero_state(batch_size, tf.float32)



                    rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell,self.rnn_inputs,sequence_length=self.steps, initial_state=self.state)
                    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=self.keepprob,name='rnn_outputs')


                    #
                    # rnn_outputs = tf.transpose(rnn_outputs,perm=[1,0,2])

                with tf.name_scope('output_layer'):
                    with tf.name_scope('Weights'):
                        Wout = self.weight_variable([self.cell_size,self.num_classes],name='W_out')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('output_layer/Weights',Wout)
                    with tf.name_scope('Biases'):
                        Bout = self.bias_variable([self.num_classes],name='B_out')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('outpu_layer/Weights',Bout)


                    if self.is_classifier is True:
                        # get last rnn output
                        idx = tf.range(tf.cast(batch_size,tf.int64))
                        idx = idx * tf.cast(tf.shape(tf.cast(rnn_outputs,tf.int64))[1],tf.int64)
                        idx = idx + (self.steps - 1)
                        rnn_outputs = tf.gather(tf.reshape(rnn_outputs, [-1, self.cell_size]), idx)
                    else:
                        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.cell_size])


                    #for seq2seq calcuations
                    self.rnn_outputs = rnn_outputs


                    logits = tf.add(tf.matmul(rnn_outputs,Wout),Bout)

                    # for seq2seq calculations
                    self.logits = logits

                    self.predictions = tf.argmax(tf.nn.softmax(logits),axis=-1)
                    self.flat_labels = tf.reshape(self.ys, [-1])


                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.flat_labels,self.predictions),tf.float32),name='accuracy')
                    tf.summary.scalar('Accuracy',self.accuracy)

                with tf.name_scope('cross_entropy'):

                    if self.is_classifier:
                        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.ys))
                    else:
                        # seq2seq loss
                        self.seq_logits =  tf.unstack(tf.reshape(self.logits, [-1,self.max_steps,self.num_classes]),axis=1)
                        self.seq_lables =  tf.unstack(self.ys,num=self.max_steps, axis=1)
                        # self.seq_lables = tf.reshape(self.ys,[-1])
                        self.seq_weights = tf.unstack(tf.sign(tf.reduce_max(tf.abs(self.xs),axis=-1)),axis=1,num=self.max_steps)

                        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.flat_labels))
                        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.seq_logits,self.seq_lables,self.seq_weights,name='loss')
                    tf.summary.scalar('Cross Entropy', self.loss)
                    # weights = tf.reshape
                    # tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits=logits, labels=self.logits)
                if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                    with tf.name_scope('train'):
                        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step,name='train_step')

                self.saver = tf.train.Saver(tf.global_variables(),keep_checkpoint_every_n_hours=1,max_to_keep=2)

                self.g_init = tf.global_variables_initializer()
                self.l_init = tf.local_variables_initializer()

                self.sess = tf.Session()

                self.summary = tf.summary.merge_all()

                if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                    self.summary_writer = tf.summary.FileWriter(self.logs_path + '/train' ,self.sess.graph)
                elif self.oper_mode == RNNModel.OperMode.OPER_MODE_EVAL:
                    self.summary_writer = tf.summary.FileWriter(self.logs_path + '/eval', self.sess.graph)


    def weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    def reset_graph(self):
        if self.sess is not None:
            self.sess.close()
        tf.reset_default_graph()

    def init_graph(self):


        try:
            print('Restoring Lastest Checkpoint : ',end=' ')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
            print('SUCCESS')


        except Exception as err:
            print('FAILED')
            self.sess.run(self.g_init)

        finally:
            self.sess.run(self.l_init)
            self.coord = tf.train.Coordinator()

    def train(self,keepprob=1.0):

        self.init_graph()

        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        path = os.path.join(self.model_path, self.model_name)

        try:
            feed_dict = {self.keepprob : keepprob}

            count = 0
            total_loss = 0.0
            total_acc = 0.0

            while not self.coord.should_stop():
                _,loss,  accuracy, summary, final_state, ys = self.sess.run([self.train_step,self.loss, self.accuracy, self.summary, self.final_state, self.ys],feed_dict)
                total_loss += loss
                total_acc += accuracy
                count += 1
                #_, loss, accuracy, summary, rnn_outputs, logits, predictions, flat_labels = self.sess.run(
                #    [self.train_step, self.loss, self.accuracy, self.summary, self.rnn_outputs, self.logits, self.predictions, self.flat_labels], feed_dict)
                # print('State:{} {} {}'.format(self.oper_mode,np.shape(state),state))
                # print(final_state)
                # print('Final State:{} {} {}'.format(self.oper_mode,np.shape(final_state), final_state))
                current_step = tf.train.global_step(self.sess, self.global_step)
                if current_step % self.validation_step == 0:
                    self.summary_writer.add_summary(summary, current_step)
                    self.summary_writer.flush()
                    # print(seq_weights)
                    # print('Saving model params for step: ', current_step)

                    print('{} Loss: {} Accuarcy {}'.format(self.oper_mode, total_loss / count, total_acc / count))
                    total_loss = 0.0
                    total_acc = 0.0
                    count = 0

                    self.saver.save(self.sess, path, global_step=current_step, write_meta_graph=False)
                    if self.evalfn:
                        self.evalfn(current_step)



                # print(np.shape(pred), np.shape(labels), accuracy)
                # print(pred,labels)
                # feed_dict[self.state] = final_state





        except tf.errors.OutOfRangeError:
            print('Out Of Range Errror')
            # pass
        finally:
            self.summary_writer.close()
            self.coord.request_stop()

        current_step = tf.train.global_step(self.sess, self.global_step)
        self.saver.save(self.sess, path, global_step=current_step,write_meta_graph=False)

        self.coord.join(self.threads)
        self.threads = None



    def evaluate(self, curr_step, keepprob=1.0):

        self.init_graph()

        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        count = 0
        total_loss = 0.0
        total_acc = 0.0
        summary = None
        try:
            feed_dict = {self.keepprob: keepprob}

            while not self.coord.should_stop():
                loss, accuracy, summary = self.sess.run([self.loss,self.accuracy, self.summary],feed_dict)
                total_loss += loss
                total_acc += accuracy
                count += 1




        except Exception as err:
            pass
        finally:
            self.coord.request_stop()
            if count > 0:
                self.summary_writer.add_summary(summary, curr_step)
                self.summary_writer.flush()
                print('{} Loss : {}  Accuracy: {}'.format(self.oper_mode, total_loss/count, total_acc/count))
        self.coord.join(self.threads)
        self.threads = None

    def test(self,xs,steps):
        feed_dict = {self.keepprob: 1.0, self.xs : xs, self.steps : steps}

        score = self.sess.run(self.predictions, feed_dict)

        print(score[:steps[0]])

        score = np.sum(score)/steps

        return score


    class Builder():
        def __init__(self):
            self.epochs = 1
            self.batch_size = 20
            self.read_path = ''
            self.feature_size = 13
            self.num_classes = 2
            self.cell_size = 128
            self.max_steps = 50
            self.num_layers = 1
            self.learning_rate = 1e-4
            self.logs_path = ''
            self.model_path = ''
            self.model_name = ''
            self.cell_type = RNNModel.CellType.RNN_CEL_TYPE_LSTM
            self.evalfn = None
            self.oper_mode = RNNModel.OperMode.OPER_MODE_NONE
            self.validation_step = 10
            self.is_classifier = False



        def set_epochs(self,val):
            self.epochs = val
            return self

        def set_batch_size(self,val):
            self.batch_size = val
            return self

        def set_read_path(self,val):
            self.read_path = val
            return self

        def set_feature_size(self,val):
            self.feature_size = val
            return self

        def set_class_size(self,val):
            self.num_classes = val
            return self

        def set_cell_size(self,val):
            self.cell_size = val
            return self

        def set_cell_type(self,val):
            self.cell_type = val
            return self

        def set_max_steps(self,val):
            self.max_steps = val
            return self

        def set_layer_size(self,val):
            self.num_layers = val
            return self

        def set_learning_rate(self,val):
            self.learning_rate = val
            return self

        def set_eval_fn(self,val):
            self.evalfn = val
            return self

        def set_oper_mode(self,val):
            self.oper_mode = val
            return self

        def set_validation_step(self,val):
            self.validation_step = val
            return self

        def set_logs_path(self,val):
            self.logs_path = val
            return self

        def set_model_path(self,val):
            self.model_path = val
            return self

        def set_model_name(self,val):
            self.model_name = val
            return self

        def set_classifer_status(self,flag):
            self.is_classifier = flag
            return self


        def build(self):
            return RNNModel(builder=self)


    class CellType(Enum):
        RNN_CELL_TYPE_NONE = 0
        RNN_CEL_TYPE_LSTM = 1
        RNN_CELL_TYPE_GRU = 2

    class OperMode(Enum):
        OPER_MODE_NONE = 0
        OPER_MODE_TRAIN = 1
        OPER_MODE_EVAL = 2
        OPER_MODE_TEST = 3




