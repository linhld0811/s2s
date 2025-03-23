import os
from glob import glob
from tqdm import tqdm
import librosa
from datasets import load_dataset, Dataset

def resample(data_point, target_sr=16000):

    original_array  = data_point['array']
    original_sr     = data_point['sampling_rate']
    resampled_array = librosa.resample(original_array, orig_sr=original_sr, target_sr=16000)

    resampled_data_point = {
        'array': resampled_array,
        'sampling_rate': target_sr,
    }

    return resampled_data_point

data_dir = "/root/datasets/gpt-omni/VoiceAssistant-400K"
list_file = [os.path.basename(i) for i in glob(f"{data_dir}/data/*.parquet")]
for file in tqdm(list_file):
    data = load_dataset("/root/datasets/gpt-omni/VoiceAssistant-400K/data", data_files=file)["train"]
    instruction = []
    instruction_only = []
    for d in tqdm(data):
        r = d["round"]
        if r == "None":
            index = d["index"]
        elif r == "0":
            index = d["index"] + "_" +  r
        else:
            continue
        question = d["question"]
        question_audio = d["question_audio"]
        instruction.append({"index": index, "question_audio": resample(question_audio), 
                            "question": question})
        instruction_only.append({"index": index, "question": question, "answer": d["answer"]})
    instruction = Dataset.from_list(instruction)
    instruction.to_parquet(f"datasets/VoiceAssistant-400K-Instruction/{file}")
    instruction_only = Dataset.from_list(instruction_only)
    instruction_only.to_json(f'datasets/VoiceAssistant-400K-Instruction-only/{file.replace("parquet", "json")}')
