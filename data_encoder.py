import tensorflow as tf
import features
import os,utils
import numpy as np
import re


class TFEncoder(object):
    def __init__(self,builder):
        self.src = builder.src
        self.dst = builder.dst
        self.max_steps = builder.max_steps
        self.seq_id = builder.seq_id

    def dump_record(self, ex, type):
        dst = os.path.join(self.dst,type)

        filename = os.path.join(dst, '{}.trf'.format(utils.get_file_timestamp()))
        with tf.python_io.TFRecordWriter(filename) as fp:
            fp.write(ex.SerializeToString())

    def make_seq_example(self, xs, ys):

        sequence_length = len(xs)

        xs = np.pad(xs, [[0, self.max_steps - sequence_length],[0,0]],mode='constant')
        ex = tf.train.SequenceExample()
        # add context features

        ex.context.feature['len'].int64_list.value.append(sequence_length)
        ex.context.feature['label'].int64_list.value.append(ys)

        # ex.context.feature['id'].int64_list.value.append(id)
        # ex.context.feature['seq-id'].int64_list.value.append(self.seq_id)
        # self.seq_id += 1

        # add sequence features
        fl_tokens = ex.feature_lists.feature_list['tokens']
        fl_labels = ex.feature_lists.feature_list['labels']

        labels = np.ones([len(xs)],dtype=np.int64) * ys
        labels = np.pad(labels,[[0,self.max_steps - sequence_length]],mode='constant' )

        for token, label in zip(xs, labels):
            if np.ndim(token) == 0:
                fl_tokens.feature.add().float_list.value.append(token)
            else:
                fl_tokens.feature.add().float_list.value.extend(token)

            fl_labels.feature.add().int64_list.value.append(label)


        return ex

    def encode(self,duration=None):
        max_steps = 0
        min_steps = 10000000
        file_count = 0
        exceed_count = 0
        for r, d, f in os.walk(self.src):
            for fname in f:

                file_count += 1

                filename = os.path.join(r, fname)

                xs, id = features.parse_audio_file(filename, normalize=True, duration=duration)
                steps = len(xs)
                xs = xs[:self.max_steps]

                if steps > self.max_steps:
                    exceed_count += 1

                print(file_count, max_steps, min_steps, steps)
                if steps > max_steps:
                    max_steps = steps

                if min_steps > steps:
                    min_steps = steps

                child_dir = os.path.basename(r)
                parent_dir = os.path.basename(os.path.dirname(r))



                ys = 0
                if child_dir == 'positive':
                    ys = 1

                ex = self.make_seq_example(xs, ys)

                self.dump_record(ex,parent_dir)

        return file_count, max_steps, min_steps, exceed_count


    def get_info(self,duration=None):
        max_steps = 0
        min_steps = 10000000
        file_count = 0
        exceed_count = 0
        for r, d, f in os.walk(self.src):
            for fname in f:

                file_count += 1

                filename = os.path.join(r, fname)

                xs, id = features.parse_audio_file(filename,normalize=True,duration=duration)
                steps = len(xs)
                xs = xs[:self.max_steps]

                if steps > 600:
                    exceed_count += 1


                print(file_count, max_steps, min_steps, steps)
                if steps > max_steps:
                    max_steps = steps

                if min_steps > steps:
                    min_steps = steps

        return file_count, max_steps,min_steps, exceed_count

    class Builder():
        def __init__(self):
            self.src = ''
            self.dst = ''
            self.max_steps = None
            self.seq_id = 0

        def set_src_path(self, val):
            self.src = val
            return self

        def set_dst_path(self, val):
            self.dst = val
            return self

        def set_max_steps(self,val):
            self.max_steps = val
            return self
        def set_seq_id(self,val):
            self.seq_id = val
            return self

        def build(self):
            return TFEncoder(self)






if __name__ == '__main__':



    encoder = TFEncoder.Builder().\
        set_src_path(r'data').\
        set_dst_path(r'records').\
        set_max_steps(600).\
        build()

    # result = encoder.get_info()
    # result = encoder.xencode(0,(2000, 5500))

    utils.clean_dir(r'records')
    result = encoder.encode()

    print(result)

    # encoder = TFEncoder.Builder().\
    #     set_src_path('data/eval').\
    #     set_dst_path('records/eval').\
    #     set_max_steps(50).\
    #     set_seq_id(0).\
    #     build()
    #
    # encoder.encode()











    # max_steps = 0
    # min_steps = 1000000
    # steps = len(xs)

#         print(max_steps, min_steps, steps)
#         if steps > max_steps:
#             max_steps = steps
#
#         if min_steps > steps:
#             min_steps = steps
#
#
# return max_steps,min_steps






