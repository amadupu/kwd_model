import utils
import os
import shutil
import random
import numpy as np
from pydub import AudioSegment
from data_encoder import TFEncoder

# windows
if os.name =='nt':
    noise_path = r'D:\codeathon-18\demand'
    alexa_path = r'D:\codeathon-18\alexa'
    mixed_path = r'D:\codeathon-18\mixed'
    valid_train_path = r'D:\codeathon-18\cv_corpus_v1\cv_corpus_v1\cv-valid-train'
    valid_test_path = r'D:\codeathon-18\cv_corpus_v1\cv_corpus_v1\cv-valid-test'
else:
    noise_path = r'/home/arun_madupu/projects/corpus/demand'
    alexa_path = r'/home/arun_madupu/projects/corpus/alexa'
    mixed_path = r'/home/arun_madupu/projects/corpus/mixed'
    valid_train_path = r'/home/arun_madupu/projects/corpus/cv_corpus_v1/cv-valid-train'
    valid_test_path = r'/home/arun_madupu/projects/corpus/cv_corpus_v1/cv-valid-test'


def init_data_sets(num_samples,train_ratio):

    train_samples = int(num_samples * train_ratio)
    eval_samples = num_samples - train_samples

    # clean data path
    utils.clean_dir('data')

    train_pos_samples = int(train_samples * 0.5)
    train_neg_samples = train_samples - train_pos_samples

    eval_pos_samples = int(eval_samples * 0.5)
    eval_neg_samples = eval_samples - eval_pos_samples

    # divide alexa data into train postive and eval positive as per train ratio
    distribution_ratio = int(train_ratio/(1-train_ratio))
    count  = 0
    print('Copying Alexa Base Files..')
    for r,d,f in os.walk(alexa_path):
        for file in f:
            count += 1

            filename = os.path.join(r, file)

            if count % distribution_ratio != 0:
                # copy into train / positive
                target = os.path.join('data',os.path.join('train', 'positive'))
                train_pos_samples -= 1
            else:
                # copy into eval / positive
                target = os.path.join('data',os.path.join('eval', 'positive'))
                eval_pos_samples -= 1

            shutil.copy(filename, os.path.join(target,'{}-{}'.format(os.path.basename(r),file)))


    # divide mixed data into train postive and eval positive as per train ratio
    mixed_files = list()

    for r,d,f in os.walk(mixed_path):
        for file in f:
            filename = os.path.join(r,file)
            mixed_files.append(filename)

    mixed_file_len = len(mixed_files)
    total_pos_samples_required = train_pos_samples + eval_pos_samples

    if total_pos_samples_required > mixed_file_len:
        raise Exception('Number of Mixed Samples Needed Exceed the Quota')

    train_end = int(mixed_file_len * (train_pos_samples/total_pos_samples_required))
    train_pos_indices = random.sample(range(0,train_end),train_pos_samples)
    eval_pos_indices = random.sample(range(train_end,mixed_file_len),eval_pos_samples)



    target = os.path.join('data', os.path.join('train', 'positive'))
    print('Copying Alexa + Noise Mixed Training Files: ', len(train_pos_indices))
    for index in train_pos_indices:
        filename = mixed_files[index]
        shutil.copy(filename, target)

    target = os.path.join('data', os.path.join('eval', 'positive'))
    print('Copying Alexa + Noise Mixed Eval Files: ', len(eval_pos_indices))
    for index in eval_pos_indices:
        filename = mixed_files[index]
        shutil.copy(filename, target)


    # copy the train negative samples

    file_lst = os.listdir(valid_train_path)
    file_len = len(file_lst)
    file_indices = random.sample(range(0,file_len),train_neg_samples)

    target = os.path.join('data', os.path.join('train', 'negative'))
    print('Copying Train Negative Samples: ', len(file_indices))
    for index in file_indices:
        filename = file_lst[index]
        filepath = os.path.join(valid_train_path,filename)
        if os.path.splitext(filepath)[-1] == '.mp3':
            utils.convert(filepath,os.path.join(target,'{}.wav'.format(os.path.splitext(filename)[0])),16000)
        else:
            shutil.copy(filepath, os.path.join(target,filename))

    # copy the eval negative samples

    file_lst = os.listdir(valid_test_path)
    file_len = len(file_lst)
    file_indices = random.sample(range(0,file_len),eval_neg_samples)

    target = os.path.join('data', os.path.join('eval', 'negative'))

    print('Copying Eval Negative Samples: ',len(file_indices))
    for index in file_indices:
        filename = file_lst[index]
        filepath = os.path.join(valid_test_path,filename)
        if os.path.splitext(filepath)[-1] == '.mp3':
            utils.convert(filepath,os.path.join(target,'{}.wav'.format(os.path.splitext(filename)[0])),16000)
        else:
            shutil.copy(filepath, os.path.join(target,filename))

def perform_audio_mixing(file1,file2,target):
    sound1 = AudioSegment.from_file(file1)
    sound2 = AudioSegment.from_file(file2)

    combined = sound1.overlay(sound2)

    combined.export(target, format='wav')

def create_mixed_data():
    os.makedirs(mixed_path, exist_ok=True)
    utils.clean_dir(mixed_path)
    noise_files = list()
    for r,d,f in os.walk(noise_path):
        for fs in f:
            filename = os.path.join(r,fs)
            noise_files.append(filename)


    count1 = 0
    count2 = 0
    count3 = 0
    for r,d,f in os.walk(alexa_path):
        for fs in f:
            count1 += 1
            indices = np.random.randint(0,len(noise_files),5)
            n_files = [ noise_files[i] for i in list(indices)]
            for fn in n_files:
                count2 += 1
                count3 += 1
                target = os.path.join(mixed_path,'{}.wav'.format(utils.get_file_timestamp()))
                perform_audio_mixing(os.path.join(r,fs),fn,target)
                print('Processing: {} {} {}'.format(count1,count2,count3))
            count2 = 0

def init_records():
    encoder = TFEncoder.Builder().\
        set_src_path(r'data').\
        set_dst_path(r'records').\
        set_max_steps(600).\
        build()

    utils.clean_dir(r'records')
    
    result = encoder.encode()

    print(result)

if __name__ == '__main__':
    print('CREATING NOISE VERSION OF SPEECH SAMPLES')
    create_mixed_data()
    print('INITIALIZING DATA SETS')
    init_data_sets(1250,0.8)
    print('INITIALIZING RECORDS')
    init_records()
    print('FINISHED')
