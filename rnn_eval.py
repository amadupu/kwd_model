from rnn_model import RNNModel
from pocketsphinx.pocketsphinx import *
import os
import pyaudio
import numpy as np
import time
from pydub import AudioSegment
from pydub.silence import split_on_silence
import utils
import struct
from features import get_features
from features import parse_audio_file
from paths import *

modeldir = r'D:\VM\vmshare\speech_recognition\pocketsphinx\model'
datadir = r'D:\VM\vmshare\speech_recognition\pocketsphinx\test\data'
kws_file = r'key-words.lst'
outfile = 'recording.raw'

in_speech = False
utterance = b''

silence_thresold = -40
min_silence_len=400
keep_silence=100

is_in_speech = False

chunk_count = 0
sample_width = 2
frame_rate = 16000
channels = 1
frame_width = 2

frames_per_buffer = 1024

feature_size=26
max_steps = 600
cell_type = RNNModel.CellType.RNN_CELL_TYPE_GRU
cell_size = 512
batch_size = 1
num_classes = 2
num_layers = 2
model_name = 'alexa'
is_classifer = True
model_path = 'cls_model'
time_major = False
keep_prob = 1.0

config = Decoder.default_config()
config.set_string('-hmm',os.path.join(modeldir,r'en-us\en-us'))
config.set_string('-dict', os.path.join(modeldir, r'en-us\cmudict-en-us.dict'))
config.set_string('-keyphrase', 'alexa')
config.set_float('-kws_threshold', 1e-20)
config.set_string('-logfn', 'null')

decoder = Decoder(config)
decoder.start_utt()

model = RNNModel.Builder().set_max_steps(max_steps). \
    set_feature_size(feature_size). \
    set_cell_type(RNNModel.CellType.RNN_CELL_TYPE_GRU). \
    set_cell_size(cell_size). \
    set_batch_size(batch_size). \
    set_class_size(num_classes). \
    set_layer_size(num_layers). \
    set_model_path(model_path). \
    set_model_name(model_name). \
    set_time_major(time_major).\
    set_classifer_status(is_classifer). \
    set_oper_mode(RNNModel.OperMode.OPER_MODE_TEST). \
    build()

model.init_graph()

data = b''
batch_count = 15
count = 0


def evaluator():
    for r,d,files in os.walk(alexa_path):
        for f in files:
            filename = os.path.join(r,f)
            features,_ = parse_audio_file(filename)
            sequence_length = len(features)
            delta_length = sequence_length % max_steps
            xs = np.pad(features, [[0, max_steps - delta_length], [0, 0]], mode='constant')
            # steps = np.ones(len(xs)) * 100
            # steps[-1] = delta_length
            xs = np.reshape(xs,[-1,max_steps,feature_size])
            steps = np.ones(np.shape(xs)[0],dtype=np.int64) * 100
            steps[-1] = delta_length
            # xs = np.transpose(xs,(1,0,2))
            print('file: {} input shape: {} steps: {}'.format(f,np.shape(xs),steps))
            score, logits = model.test(xs, steps)

            logits = np.reshape(logits,[max_steps,-1,3])
            logits = np.transpose(logits,(1,0,2))
            logits = np.reshape(logits,[-1,3])
            logits = np.mean(logits,axis=0)

            score = np.reshape(score,[max_steps,-1])
            score = np.transpose(score,(1,0))
            score = np.reshape(score,[-1])
            print(logits)



# evaluator()
def callback(in_data, frame_count, time_info, status):
    global data, count
    global in_speech, utterance, is_in_speech, chunk_count, model

    # count += 1

    # outstream.write(in_data)
    # if count % batch_count != 0:
    #     data += in_data
    #     return (None, pyaudio.paContinue)
    #
    # data += in_data
    #
    #
    #
    # fmt = '<%ih' % (len(data) / sample_width)
    # signal = np.array(struct.unpack_from(fmt, data))
    # xs = get_features(signal, frame_rate, True)
    # steps = len(xs)
    # xs = xs[:max_steps]
    # sequence_length = len(xs)
    #
    #
    # print('Sequence Length: {}  in_data: {} data: {}'.format(sequence_length,len(in_data),len(data)))
    # data = b''
    # xs = np.pad(xs, [[0, max_steps - sequence_length], [0, 0]], mode='constant')
    #
    # input = np.reshape(xs, [-1, max_steps, feature_size])
    #
    # input = np.transpose(input,(1,0,2))
    #
    # score,logits = model.test(input, [sequence_length])
    #
    # logits = np.mean(logits,axis=0)
    # # print(score)
    # print(logits)



    decoder.process_raw(in_data, False, False)

    if decoder.get_in_speech():
        if is_in_speech is False:
            is_in_speech = True
            chunk_count += 1

        utterance += in_data
        # print('IN SPEECH: ', len(utterance))
    else:
        # print('SILENCE : ',len(utterance))
        if is_in_speech:
            is_in_speech = False
            metadata = {
                'sample_width': sample_width,
                'frame_rate': frame_rate,
                'channels': channels,
                'frame_width': frame_width
            }
            audio_segment = AudioSegment(utterance,metadata=metadata)
            audio_chunks = split_on_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresold, keep_silence=keep_silence)


            # data = audio_segment.raw_data
            # fmt = '<%ih' % (len(data) / sample_width)
            # signal = np.array(struct.unpack_from(fmt, data))
            # xs = get_features(signal, frame_rate, True)
            # steps = len(xs)
            # xs = xs[:max_steps]
            # sequence_length = len(xs)
            # xs = np.pad(xs, [[0, max_steps - sequence_length], [0, 0]], mode='constant')
            # pred,logits = model.test(np.reshape(xs, [-1, max_steps, feature_size]), [sequence_length])
            #
            # print(pred)
            # print(logits)

            for i, chunk in enumerate(audio_chunks):
                data = chunk.raw_data
                fmt = '<%ih' % (len(data) / sample_width)
                signal = np.array(struct.unpack_from(fmt, data))
                xs = get_features(signal,frame_rate,True)
                steps = len(xs)
                xs = xs[:max_steps]
                sequence_length = len(xs)
                xs = np.pad(xs, [[0, max_steps - sequence_length], [0, 0]], mode='constant')
                score,logits = model.test(np.reshape(xs,[-1,max_steps,feature_size]),[sequence_length])

                if score[0] == 1:
                    print('Detected')
                else:
                    print('Not Detected')

                # print('segment: {} {} score: {}'.format(chunk_count,i,score))


                #
                # out_file = os.path.join('chunks',
                #                         'segment-{}-{}-{}-{}.raw'.format(chunk_count, i,steps,score))
                # chunk.export(out_file, format='raw')




        # process utterance
        utterance = b''


    return (None, pyaudio.paContinue)




p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=channels, rate=frame_rate, input=True, frames_per_buffer=frames_per_buffer,stream_callback=callback)
outstream = p.open(format=pyaudio.paInt16, channels=channels, rate=frame_rate, output=True, frames_per_buffer=frames_per_buffer)

stream.start_stream()
outstream.start_stream()
while True:
    time.sleep(1)

print('Shutting down')
fp.close()
stream.close()
p.terminate()
