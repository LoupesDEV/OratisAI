from jiwer import wer, cer
import os

def evaluate_results(decoded_dir, reference_dir):
    decoded_files = [f for f in os.listdir(decoded_dir) if f.endswith(".txt")]
    total_wer = 0
    total_cer = 0
    count = 0

    for file_name in decoded_files:
        decoded_path = os.path.join(decoded_dir, file_name)
        reference_path = os.path.join(reference_dir, file_name.replace("_processed", ""))

        with open(decoded_path, "r") as d_file, open(reference_path, "r") as r_file:
            decoded_text = d_file.read().strip()
            reference_text = r_file.read().strip()
            total_wer += wer(reference_text, decoded_text)
            total_cer += cer(reference_text, decoded_text)
            count += 1

    avg_wer = total_wer / count
    avg_cer = total_cer / count
    print(f"WER: {avg_wer:.2f}, CER: {avg_cer:.2f}")

if __name__ == "__main__":
    evaluate_results("results/decoded_texts", "data/transcripts")