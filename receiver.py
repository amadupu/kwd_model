import socket
import pyaudio
import time
import utils
import librosa
from pydub import AudioSegment
import threading
from queue import Queue


queue = Queue()
pqueue = Queue()

UDP_IP = ''
UDP_PORT = 5005
PLAY_UDP_PORT = 5006
channels = 1
frame_width = 2
sample_width = 2
frame_rate = 16000
frames_per_buffer = 1024


def play_thread():
    global pqueue, stream

    # print('PLAY THREAD: Play Thread Running')

    while True:
        data = pqueue.get(block=True)
        print('PLAY THREAD: Playing')
        stream.write(data)



def receive_thread():
    global pthread, pqueue
    # print('RECV THREAD: Staring Play Thread')
    pthread.start()
    while True:
        data, addr = psock.recvfrom(6000)  # buffer size is 1024 bytes
        # print('RECV THREAD: Posting to play thread')
        pqueue.put(data, block=False)


def child_process():


    metadata = {
        'sample_width': sample_width,
        'frame_rate': frame_rate,
        'channels': channels,
        'frame_width': frame_width
    }


    while True:

        data = queue.get(block=True)
        seg = AudioSegment(data, metadata=metadata)
        # print("UPSAMP: received message:", len(data),len(seg.raw_data))


        seg = seg.set_frame_rate(44100)
        print("UPSAMP: Post upsampling:", len(data), len(seg.raw_data))

        sock.sendto(seg.raw_data,('172.24.150.50',5001))



        # resample
        # stream.write(seg.raw_data)


sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

psock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
psock.bind((UDP_IP, PLAY_UDP_PORT))



p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=channels, rate=44100, output=True,
                frames_per_buffer=1024)

stream.start_stream()


pthread = threading.Thread(target=play_thread, args=(), daemon=False)


thread = threading.Thread(target=child_process, args=(), daemon=False)
thread.start()

rthread = threading.Thread(target=receive_thread, args=(), daemon=False)
rthread.start()



while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    queue.put(data,block=False)

thread.join()
rthread.join()
pthread.join()


    # print("received message:", len(data))
