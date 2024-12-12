import os
import argparse
from pydub import AudioSegment

def run(config, segment_audio):
    input_dir = config['paths']['raw_audio']
    output_dir = config['paths']['processed_audio']
    segment_length = 30 * 1000  # 30 secondes en millisecondes

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.wav', '.mp3')):
            audio_path = os.path.join(input_dir, file_name)
            audio = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(16000)

            if segment_audio:
                for i in range(0, len(audio), segment_length):
                    segment = audio[i:i+segment_length]
                    segment_name = f"{os.path.splitext(file_name)[0]}_seg{i // segment_length}.wav"
                    segment_path = os.path.join(output_dir, segment_name)
                    segment.export(segment_path, format="wav")
            else:
                full_output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.wav")
                audio.export(full_output_path, format="wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement des fichiers audio")
    parser.add_argument("--segment", choices=["yes", "no"], default="yes", help="Segmenter les fichiers audio en morceaux de 30 secondes")

    args = parser.parse_args()
    segment_audio = args.segment == "yes"

    config = {
        'paths': {
            'raw_audio': "data/raw",
            'processed_audio': "data/processed"
        }
    }

    run(config, segment_audio)