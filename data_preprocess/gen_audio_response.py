import json
import random
import hashlib
from tqdm import tqdm
import soundfile as sf
import torch
from datasets import load_dataset, Dataset
from kokoro import KPipeline

def gen_index(text, length=16, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"):
    seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (10**9) 
    random.seed(seed) 
    sequence = ''.join(random.choice(alphabet) for _ in range(length))
    return sequence


pipeline = KPipeline(lang_code='a')
data = load_dataset("datasets", data_files="alpaca_dataset_response.json")["train"]

done = []
try:
    c = open("datasets/tatsu-lab/alpaca_dataset_response_w_audio.json", "r")
    for line in c.readlines():
        done.append(json.loads(line.strip())["instruction"])
    c.close()
except:
    pass

remain = [i for i in data if i["instruction"] not in done]
print(len(remain), data.num_rows)


done = load_dataset("datasets", data_files="alpaca_dataset_response_w_audio.json")["train"]
new_data = [i for i in done]
for i in tqdm(remain):
    text = i["response"]
    if len(text.split(" ")) > 70:
        continue
    index = gen_index(text)
    generator = pipeline(
        text, voice='af_heart', 
        speed=1, split_pattern=r'\n+'
    )
    wav = torch.tensor([])
    for gs, ps, audio in generator:
        wav = torch.concat((wav, audio), dim=0)
    sf.write(f'wavs/{index}.wav', audio, 24000)
    i["index"] = index
    new_data.append(i)
new_data = Dataset.from_list(new_data).to_json("datasets/alpaca_dataset_response_w_audio.json")
