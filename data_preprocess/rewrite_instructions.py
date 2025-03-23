import json

from datasets import load_dataset

import multiprocessing
import time
import itertools
from collections import defaultdict

from google import genai
from google.genai import types
from langchain_core.output_parsers import JsonOutputParser

instruction_rewriting_prompt = """Below is an instruction data containing the user’s instruction. I would like to generate a speech version of this instruction for training a large language model that supports speech input. Therefore, please rewrite my instruction data according to the following requirements:
1. Modify the instruction to simulate human speech, adding fillers as appropriate (but not too many ‘you
know’, ‘like’, etc.).
2. The question should not contain content that cannot be synthesized by the TTS model. Numbers should be written in English words rather than Arabic numerals.
3. The question should be relatively brief without excessive verbiage.
4. Not using Okay word in starting sentences.
[instruction]: {instruction}
Please output in JSON format as follows: {{"instruction_rewrite": question}}."""


#gemma-3-27b-it gemini-2.0-flash-lite
def process_item(item, key):
    parser = JsonOutputParser()
    try:
        print(f"Processing item: {item} with key: {key}")

        client = genai.Client(
            api_key=key,
            )
        response = client.models.generate_content(
            model='gemma-3-27b-it', contents=instruction_rewriting_prompt.format(instruction=item["instruction"]), 
                config=types.GenerateContentConfig(
                    temperature=0,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE",
                        ),
                    ]
                ),
            )
        item["instruction_rewrite"] = parser.parse(response.text)["instruction_rewrite"]
        with open(f"{data_dir}/ultrachat_first_turn_rewrite.json", "a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        time.sleep(1)
        return item, True
    except Exception as e:
        print(f"Error processing item {item} with key {key}: {e}")
        return item, False

def process_chunk(items_with_keys, results_queue):
    """
    Processes a chunk of items with a rate limit and keys.
    """
    processed_count = 0
    start_time = time.time()
    for item, key in items_with_keys:
        result, is_done = process_item(item, key)
        results_queue.put((result, key, is_done))
        processed_count += 1

        if processed_count % 5 == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time < 10:
                time.sleep(10 - elapsed_time)
            start_time = time.time()

def dump_json_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(item_list, keys_list):
    num_processes = len(keys_list)
    chunk_size = len(item_list) // num_processes
    chunks = [item_list[i:i + chunk_size] for i in range(0, len(item_list), chunk_size)]

    key_cycle = itertools.cycle(keys_list)
    items_with_keys = []
    for chunk in chunks:
        items_with_keys.append(list(zip(chunk, [next(key_cycle)] * len(chunk))))

    results_queue = multiprocessing.Queue()
    processes = []

    for chunk_with_keys in items_with_keys:
        process = multiprocessing.Process(target=process_chunk, args=(chunk_with_keys, results_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    results = []
    key_success_counts = defaultdict(int)
    key_failure_counts = defaultdict(int)
    while not results_queue.empty():
        result, key, success = results_queue.get()
        if success:
            key_success_counts[key] += 1
        else:
            key_failure_counts[key] += 1
        results.append(result)

    print(key_failure_counts)
    print(key_success_counts)
    print("All processes completed.")
    

if __name__ == "__main__":
    data_dir = "datasets/stingning"
    data = load_dataset(data_dir, data_files="ultrachat_first_turn.json")["train"]
    # data = load_dataset("datasets/tatsu-lab/alpaca")["train"]
    text_instructions = []
    for i in data:
        if len(i["instruction"].split(' ')) > 25: 
            continue
        text_instructions.append({"instruction": i["instruction"]})

    done = []
    try:
        c = open(f"{data_dir}/ultrachat_first_turn_rewrite.json", "r")
        for line in c.readlines():
            done.append(json.loads(line.strip())["instruction"])
        c.close()
    except:
        pass

    remain = [i for i in text_instructions if i["instruction"] not in done]
    print(len(remain), len(text_instructions))
   
    keys_list = ["AIzaSyCotMGzhMRTpbSKupucQK2RCSDv2n737q4", "AIzaSyC7dspQ5tdq3wls3SMD2m8i-Qm7U5HxkBo", "AIzaSyAoL1WfyGYig9J8-cMyuP8gZnvRtqSLptY"] ##, "AIzaSyCj3KrZuv-adq0PJrWX9mrL_AJAFlbqXrI", "AIzaSyDShvcTEITKYOg-GBc6MrIOcejoGRBnmdM", "AIzaSyA1t1gbBP1jcD5hfgk2QCorZNtn9vOW7Ys","AIzaSyAmqpWK3R3EsZl-88smB9mlWcCkgXYyeLE"]
    # keys_list = ["AIzaSyBQxLBV979g39IjVGaoNr2ieruMU8nSnG0","AIzaSyDGwbccoNMjlPbgSfyvur7MMk8a1Gf6vwo","AIzaSyAUBViaoZNzIzzvqG8bVNuOqjYDE_p2aGI","AIzaSyDhWvjPMJs7w4g8t-Tj0ty9tgotL3hxqps","AIzaSyAxsW6O8-E9_JUT38qKDRj6CwDH3OwJIkE","AIzaSyAabSu4IJvCc1EFsnKJiQPXIHZvQC58_hE","AIzaSyDKElwFyffOhsLY3e7AKZ4ftydGMXroVcc","AIzaSyCyVVfVdpwGB1ucNQDHDvb7V_JhUvhm2IU"]
    # keys_list = ["AIzaSyBBzWROyzCPGFpCef0J-yujjri2HEsmJs0", "AIzaSyBf_4E1RSm_9ABAfUaqW6C6HMNKXoN-_Vs", ]
    main(remain, keys_list)
