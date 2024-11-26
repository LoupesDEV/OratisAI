import os
from pydub import AudioSegment

def preprocess_audio(input_dir, output_dir, target_rate=16000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav") or file_name.endswith(".mp3"):
            audio_path = os.path.join(input_dir, file_name)
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(target_rate)
            processed_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_processed.wav")
            audio.export(processed_path, format="wav")
            print(f"Processed: {processed_path}")

if __name__ == "__main__":
    preprocess_audio("data/raw", "data/processed")