from pydub import AudioSegment
import os

def mp3_to_wav(input_path, output_folder):
    # Upload mp3 file
    audio = AudioSegment.from_mp3(input_path)

    # 원하는 길이로 분할합니다.
    # segment_length = 38 * 1000  # 10초마다 분할하도록 설정 (10,000 milliseconds)
    segments = []
    segments.append(audio[0:3800])
    segments.append(audio[3800:11800])
    segments.append(audio[11800:16400])
    segments.append(audio[16400:29100])
    segments.append(audio[29100:])
    # segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]

    # 출력 폴더가 없으면 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # wav 파일로 저장합니다.
    for i in range(0, len(segments)):
        output_path = os.path.join(output_folder, f'missile_{i + 1}.wav')
        segments[i].export(output_path, format='wav')
        print(f'Segment {i + 1} saved to {output_path}')

# 사용 예시
mp3_file_path = 'test_missile.mp3'
output_folder_path = 'record/missile'

mp3_to_wav(mp3_file_path, output_folder_path)