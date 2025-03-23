import os
import subprocess

def convert_flac_to_wav(flac_file, wav_file, sample_rate=16000):
    try:
        subprocess.run(['ffmpeg', '-i', flac_file, '-ar', str(sample_rate), wav_file], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {flac_file}: {e.stderr}")
        return False

def process_flac_files(root_dir):
    for root, _, files in os.walk(root_dir):
        if len(files) == 0: continue
        text_file = [i for i in files if i.endswith(".txt")][0]
        f = open(os.path.join(root, text_file), "r")
        fns = [line.strip().split()[0] for line in f.readlines() if len(line.strip().split(" ")) > 10]
        print(len(fns), len(files))
        for file in files:
            filename = file.split(".flac")[0]
            if file.endswith(".flac") and filename in fns:
                flac_file_path = os.path.join(root, file)

                wav_file_path = os.path.join("wavs", f"{filename}.wav")
                if convert_flac_to_wav(flac_file_path, wav_file_path):
                    continue
                else:
                    print(f"Conversion of {flac_file_path} failed.")

root_directory = "dev-clean"

process_flac_files(root_directory)
