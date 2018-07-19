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

modeldir = r'D:\VM\vmshare\speech_recognition\pocketsphinx\model'
datadir = r'D:\VM\vmshare\speech_recognition\pocketsphinx\test\data'
kws_file = r'key-words.lst'
outfile = 'recording.raw'

in_speech = False
utterance = b''

silence_thresold = -30
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
    set_classifer_status(is_classifer). \
    set_oper_mode(RNNModel.OperMode.OPER_MODE_TEST). \
    build()

model.init_graph()

def callback(in_data, frame_count, time_info, status):
    global in_speech, utterance, is_in_speech, chunk_count, model

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
            for i, chunk in enumerate(audio_chunks):
                data = audio_segment.raw_data
                fmt = '<%ih' % (len(data) / sample_width)
                signal = np.array(struct.unpack_from(fmt, data))
                xs = get_features(signal,frame_rate,True)
                steps = len(xs)
                xs = xs[:max_steps]
                sequence_length = len(xs)
                xs = np.pad(xs, [[0, max_steps - sequence_length], [0, 0]], mode='constant')
                score = model.test(np.reshape(xs,[-1,max_steps,feature_size]),[sequence_length])

                print('segment: {} {} score: {}'.format(chunk_count,i,score))



                out_file = os.path.join('chunks',
                                        'segment-{}-{}-{}-{}.raw'.format(chunk_count, i,steps,score))
                audio_segment.export(out_file, format='raw')
                break




        # process utterance
        utterance = b''


    return (None, pyaudio.paContinue)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=channels, rate=frame_rate, input=True, frames_per_buffer=frames_per_buffer,stream_callback=callback)

stream.start_stream()
while True:
    time.sleep(1)

print('Shutting down')
fp.close()
stream.close()
p.terminate()
