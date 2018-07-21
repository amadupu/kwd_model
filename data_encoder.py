import tensorflow as tf
import features
import os,utils
import numpy as np
import re
from pydub.silence import detect_nonsilent
from pydub import AudioSegment


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

    def make_seq_example(self, xs, ys,labels,id):

        sequence_length = len(xs)

        # xs = np.pad(xs, [[0, self.max_steps - sequence_length],[0,0]],mode='constant')
        ex = tf.train.SequenceExample()
        # add context features

        ex.context.feature['len'].int64_list.value.append(sequence_length)
        ex.context.feature['label'].int64_list.value.append(ys)

        ex.context.feature['id'].int64_list.value.append(id)
        ex.context.feature['seq-id'].int64_list.value.append(self.seq_id)
        self.seq_id += 1

        # add sequence features
        fl_tokens = ex.feature_lists.feature_list['tokens']
        fl_labels = ex.feature_lists.feature_list['labels']

        # labels = np.ones([len(xs)],dtype=np.int64) * ys
        # labels = np.pad(labels,[[0,self.max_steps - sequence_length]],mode='constant' )

        for token, label in zip(xs, labels):
            if np.ndim(token) == 0:
                fl_tokens.feature.add().float_list.value.append(token)
            else:
                fl_tokens.feature.add().float_list.value.extend(token)

            fl_labels.feature.add().int64_list.value.append(label)


        return ex

    def encode(self,duration=None):
        # max_steps = 0
        # min_steps = 10000000
        file_count = 0
        exceed_count = 0
        for r, d, f in os.walk(self.src):
            for fname in f:

                file_count += 1

                filename = os.path.join(r, fname)

                xs, id = features.parse_audio_file(filename, normalize=True, duration=duration)
                steps = len(xs)
                count = 0

                child_dir = os.path.basename(r)
                parent_dir = os.path.basename(os.path.dirname(r))
                target = os.path.join(self.dst, parent_dir)
                target = os.path.join(target, '{}.trf'.format(utils.get_file_timestamp()))
                tf_writer = tf.python_io.TFRecordWriter(target)

                segment = AudioSegment.from_wav(filename)
                speech_ranges = detect_nonsilent(audio_segment=segment, min_silence_len=100,
                                                 silence_thresh=-40)

                print('SPEECH: filename : {} {}'.format(fname, speech_ranges))

                while count < steps:
                    _xs = xs[count  : count + self.max_steps]
                    max_steps = len(_xs)
                    labels = np.zeros(max_steps,dtype=np.int64)
                    label = 0
                    if child_dir == 'positive':
                        label = 1

                    for start, end in speech_ranges:
                        print('SPEECH: filename : {} {} {}'.format(fname, start, end))

                        start = (int)(start / 10)
                        end = (int)(end / 10)

                        print('SPEECH WINDOW START: start: {} end: {} count: {}'.format(start,end,count))


                        if start < count:
                            if count < end:
                                start = count
                            else:
                                print('SPEECH WINDOW IGNORE: start: {} end: {} count: {}',start,end,count)
                                continue

                        print('SPEECH WINDOW END: start: {} end: {} count: {}'.format(start, end, count))

                        # adjust start and end to relative reference
                        start -= count
                        end -= count

                        print('SPEECH RELATIVE INDEX: start: {} end: {} count: {}', start, end, count)

                        # if start is falling outside the window we need to break
                        if start >= max_steps:
                            print('SPEECH START EXCEED: filename : fname: {} start: {} end: {} max_steps: {} count: {}'.format(fname, start, end, max_steps,count))
                            break

                        # if end is falling outside the window we need truncate and capture the leftover offset for next iteration
                        if end > max_steps:
                            end = max_steps
                            print('SPEECH END LIMIT: filename : fname: {} start: {} end: {} max_steps: {} count: {} end_mark: {}'.format(fname, start, end, max_steps, count, count + end))



                        if label == 0:
                            labels[start:end:] = 2
                        else:
                            labels[start:end:] = 1


                        if end == max_steps:
                            # print('SPEECH END EXCEED: filename : fname: {} start: {} end: {} max_steps: {} count: {}'.format(fname, start, end, max_steps, count))
                            break


                    print('SPEECH LABELS: filename : {} {}'.format(fname, labels))

                    ex = self.make_seq_example(_xs, label, labels,id)

                    tf_writer.write(ex.SerializeToString())

                    # move the count by max_steps captured in this iteration
                    count += max_steps

                tf_writer.close()
        return file_count


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
        set_max_steps(100).\
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






