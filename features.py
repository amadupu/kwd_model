from python_speech_features import mfcc,base
import scipy.io.wavfile as wav
import os, struct
import numpy as np

sample_rate = 16000


file_map = dict()

file_id_generator = 0

def get_features(signal,rate, normalize):


    # extract features
    features = mfcc(signal,rate,winlen=0.020,winstep=0.010)

    # print('mfcc: ' , np.shape(features))

    delta = base.delta(features,5)

    features =  np.concatenate((features,delta),axis=-1)

    print('delta: ', np.shape(features))



    # perform normalization if asked to
    if  normalize is True:
        mean_features = np.mean(features,axis=0)
        var_features = np.var(features,axis=0)
        features = (features - mean_features)/var_features

    return features



def parse_audio_file(filename,normalize=True,duration=None):
    global file_id_generator


    if duration is not None:
        min_d, max_d = duration
        duration = np.random.randint(min_d,max_d)


    if filename not in file_map:
        file_map[filename] = file_id_generator
        file_id_generator = file_id_generator + 1

    if os.path.splitext(filename)[-1][1:] == 'wav':
        rate,signal = wav.read(filename)
    else:
        with open(filename, 'rb') as fp:
            # read data as signed short little endian format
            rate = sample_rate
            signal = fp.read()
            fmt = '<%ih' % (len(signal)/2)
            signal = np.array(struct.unpack_from(fmt,signal))

    if duration is not None:
        sample_length = (int(sample_rate/1000) * duration)
        if sample_length > len(signal):
            sample_length = len(signal)

        signal = signal[:sample_length]

    return get_features(signal,rate,normalize) , file_map[filename]






