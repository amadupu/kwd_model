import utils
import os

if __name__ == '__main__':

    # initialize directory structure

    os.makedirs(r'data/eval/positive',exist_ok=True)
    os.makedirs(r'data/eval/negative',exist_ok=True)
    os.makedirs(r'data/train/positive',exist_ok=True)
    os.makedirs(r'data/train/negative',exist_ok=True)
    os.makedirs(r'data/eval/positive',exist_ok=True)
    os.makedirs(r'records/eval',exist_ok=True)
    os.makedirs(r'records/train',exist_ok=True)
    os.makedirs(r'rnn_seq_model',exist_ok=True)
    os.makedirs(r'rnn_cls_model',exist_ok=True)
    os.makedirs(r'rnn_seq_logs',exist_ok=True)
    os.makedirs(r'rnn_cls_logs',exist_ok=True)

    utils.clean_dir('records')
    utils.clean_dir('data')

    # create directory