

import time
import datetime
import os
from pydub import AudioSegment


def get_file_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S%f')


def convert(source,target,sample_rate):
    seg = AudioSegment.from_mp3(source)
    seg = seg.set_channels(1)
    seg = seg.set_frame_rate(sample_rate)
    seg = seg.set_sample_width(2)
    seg.export(target,format='wav')



def resample(source,target,sample_rate):
    seg = AudioSegment.from_wav(source)
    seg = seg.set_channels(1)
    seg = seg.set_frame_rate(sample_rate)
    seg = seg.set_sample_width(2)
    seg.export(target,format='wav')


def clean_dir(path):
    for r,d,f in os.walk(path):
        for filename in f:
            filename = os.path.join(r,filename)
            if os.path.isfile(filename):
                os.remove(filename)




def detect_leading_silence(sound, silence_threshold=-25.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

