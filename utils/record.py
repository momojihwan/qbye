import pyaudio
import wave

def record():
    chunk = 1000
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 16000
    seconds = 2
    filename = "record/example.wav"

    p = pyaudio.PyAudio()

    print('RECORDING...')

    stream = p.open(format = sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    # Store data in chunks for x seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PortAudio interface
    p.terminate()

    print("FINISHED")

    # Save record data as a wave file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()