from jiwer import wer, cer
import os
import logging

def run(config):
    decoded_dir = config['paths']['decoded_texts']
    reference_dir = config['paths']['transcripts']

    decoded_files = [f for f in os.listdir(decoded_dir) if f.endswith(".txt")]
    if len(decoded_files) == 0:
        logging.error("No files found for evaluation.")
        return

    total_wer = 0
    total_cer = 0
    count = 0

    for decoded_file in decoded_files:
        decoded_path = os.path.join(decoded_dir, decoded_file)
        reference_file = decoded_file
        reference_path = os.path.join(reference_dir, reference_file)

        if not os.path.exists(reference_path):
            logging.warning(f"Missing reference for {decoded_file}, skipped.")
            continue

        with open(decoded_path, "r") as d_file, open(reference_path, "r") as r_file:
            decoded_text = d_file.read().strip()
            reference_text = r_file.read().strip()
            
            if not decoded_text or not reference_text:
                logging.warning(f"Empty decoded or reference text for {decoded_file}, skipped.")
                continue

            total_wer += wer(reference_text, decoded_text)
            total_cer += cer(reference_text, decoded_text)
            count += 1

    if count > 0:
        avg_wer = total_wer / count
        avg_cer = total_cer / count
        logging.info(f"WER: {avg_wer:.2f}, CER: {avg_cer:.2f}")
    else:
        logging.error("No files processed for evaluation.")
