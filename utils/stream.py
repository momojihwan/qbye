from scipy.io import wavfile
from scipy import signal
from queue import Queue
import numpy as np
import pyaudio
import random
import torch
import time
import math
import sys
import io
import os
import wave
# from playsound import playsound
import utils
import models


class stream_audio():
    def __init__(self,
                args,
                model,
                sample_rate=16000,
                chunk_duration=0.25,
                feed_duration=1.0,
                channels=1,
                pause=False):
        
        self.args = args
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.feed_duration = feed_duration
        self.channels = channels

        self.device_sample_rate = 44100
        self.chunk_samples = int(self.device_sample_rate * self.chunk_duration)
        self.feed_samples = int(self.device_sample_rate * self.feed_duration)

        # Queue to communicate between the audio callback and main thread
        self.q = Queue()

        # Data buffer for the input wavform
        self.data = np.zeros(self.feed_samples, dtype='int16')

        self.stream = True
        self.pause = False
        self.model = model

    def run(self):
        '''
        Define custom keyword (anchor)
        '''

        self.path = []
        # 1. Directly recoding wav files num_anchor times
        if self.num_anchor is not None:
            print("FIRST: Say your keyword {} times".format(self.args.num_anchor))    
            for i in range(3): #(self.args.num_anchor):
                
                input("Press enter to record : ")
                utils.record()
                _, signal = wavfile.read('record/record/example.wav')
                mel, _ = utils.fbank(signal, samplerate=16000,
                                     winlen=0.025, winstep=0.012, nfilt=64)    
                self.path.append(mel)
                
        
        # 2. Already have recorded wav files
        else:
            print("FIRST: Say your keyword {} times".format("3"))
            for i in range(3): #(self.args.num_anchor):
                
                input("Press enter to record : ")
                utils.record()    
                ex_path = "record/example.wav"
                self.path.append(ex_path)

        self.matrix_learned_phoneme, self.matrix_w, self.matrix_probs = utils.create_database_ctc(self.path)
        
        self.lower_boundary, self.upper_boundary = models.CTC.threshold_ctc(self.matrix_learned_phoneme, self.matrix_w, self.matrix_probs)
        print("Confidence interval: ", self.lower_boundary, self.upper_boundary)
        '''
        NEXT: Streaming
        '''

        # Callback method
        def audio_callback(in_data, frame_count, time_info, status):
            data_ = np.frombuffer(in_data, dtype='int16')
            self.data = np.append(self.data, data_)
            if len(self.data) > self.feed_samples:
                self.data = self.data[-self.feed_samples:]
                # Process data async by sending a queue.
                self.q.put(self.data)

            return (in_data, pyaudio.paContinue)

        # Open port audio
        self.audio = pyaudio.PyAudio()
        self.stream_in = self.audio.open(input=True, output=False,
                                         format=pyaudio.paInt16,
                                         channels=self.channels,
                                         rate=self.device_sample_rate,
                                         frames_per_buffer=self.chunk_samples,
                                         stream_callback=audio_callback)

        try:
            while self.stream:
                data = self.q.get()
                file_name = "record/now.wav"

                path_wavfile = utils.save_wavefile(data=data,
                                                   file_name = file_name,
                                                   stream = self.audio,
                                                   channels = self.channels,
                                                   sample_format = pyaudio.paInt16,
                                                   sample_rate = self.sample_rate)
                threshold = models.CTC.score_ctc(path= path_wavfile,
                                            matrix_learned_phoneme = self.matrix_learned_phoneme,
                                            matrix_w = self.matrix_w)
                
                print("Threshold: ", threshold)
                if threshold > self.upper_boundary:
                    # playsound('record/tingting.wav')
                    print("Predict: Keyword")
                else:
                    print("Predict: Non-keyword")

        except (KeyboardInterrupt, SystemExit):
            self.stream = False
            self.stream_in.stop_stream()
            self.stream_in.close()
            self.audio.terminate()


        

