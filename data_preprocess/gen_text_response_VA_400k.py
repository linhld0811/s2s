import json, os

from datasets import load_dataset

import multiprocessing
import time
import itertools
from collections import defaultdict
from glob import glob
from tqdm import tqdm
from google import genai
from google.genai import types
from langchain_core.output_parsers import JsonOutputParser

response_generation_prompt = """Below is the transcribed text of a user’s speech query. Please provide a response to this question, which will be converted to speech using TTS. Please follow these requirements for your response:
1. Your response should not contain content that cannot be synthesized by the TTS model, such as parentheses, ordered lists, etc. Numbers should be written in English words rather than Arabic numerals.
2. Your response should be very concise and to the point, avoiding lengthy explanations.
[instruction]: {instruction}
Please output in JSON format as follows: {{"response": response}}."""


response_rewriting_prompt = """Below is an response of the large language model assistant. I would like to generate a speech version of this response. Therefore, please rewrite my response data according to the following requirements:
1. The response should not contain content that cannot be synthesized by the TTS model. Numbers should be written in English words rather than Arabic numerals.
2. The response should be relatively brief without excessive verbiage.
[response]: {response}
Please output in JSON format as follows: {{"response_rewrite": response}}."""


#gemma-3-27b-it
def process_item(item, key, filename):
    parser = JsonOutputParser()
    try:
        print(f"Processing item: {item} with key: {key}")

        client = genai.Client(
            api_key=key,
            )
        
        response = client.models.generate_content(
            model='gemma-3-27b-it', contents=response_generation_prompt.format(instruction=item["question"]), 
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=100,
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
        new_item = {}
        new_item["index"] = item["index"]
        new_item["instruction"] = item["question"]
        new_item["response"] = parser.parse(response.text)["response"]
        with open(f"datasets/VoiceAssistant-400K-Instructions-response/{filename}", "a") as f:
            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
        time.sleep(1)
        return new_item, True
    except Exception as e:
        print(f"Error processing item {item} with key {key}: {e}")
        return item, False

def process_chunk(items_with_keys, results_queue, filename):
    """
    Processes a chunk of items with a rate limit and keys.
    """
    processed_count = 0
    start_time = time.time()
    for item, key in items_with_keys:
        result, is_done = process_item(item, key, filename)
        results_queue.put((result, key, is_done))
        processed_count += 1

        if processed_count % 30 == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                time.sleep(60 - elapsed_time)
            start_time = time.time()

def dump_json_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main(item_list, keys_list, filename):
    num_processes = len(keys_list)
    chunk_size = len(item_list) // num_processes
    chunks = [item_list[i:i + chunk_size] for i in range(0, len(item_list), chunk_size)]

    key_cycle = itertools.cycle(keys_list)
    items_with_keys = []
    for chunk in chunks:
        items_with_keys.append(list(zip(chunk, [next(key_cycle)] * len(chunk))))

    results_queue = multiprocessing.Queue()
    processes = []

    for chunk_with_keys in tqdm(items_with_keys):
        process = multiprocessing.Process(target=process_chunk, args=(chunk_with_keys, results_queue, filename))
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
    list_file = [os.path.basename(i) for i in glob("datasets/VoiceAssistant-400K-Instructions-only/*.json")]

    for filename in list_file:
        data = load_dataset("datasets/VoiceAssistant-400K-Instructions-only", data_files=filename)["train"]
        text_instructions = [i for i in data]

        done = []
        try:
            c = open(f"datasets/VoiceAssistant-400K-Instructions-response/{filename}", "r")
            for line in c.readlines():
                done.append(json.loads(line.strip())["instruction"])
            c.close()
        except:
            pass

        remain = [i for i in text_instructions if i["question"] not in done]
        print(len(remain), len(text_instructions))

        keys_list = ["AIzaSyBBzWROyzCPGFpCef0J-yujjri2HEsmJs0", "AIzaSyCj3KrZuv-adq0PJrWX9mrL_AJAFlbqXrI", "AIzaSyCotMGzhMRTpbSKupucQK2RCSDv2n737q4"]#"AIzaSyA1t1gbBP1jcD5hfgk2QCorZNtn9vOW7Ys", "AIzaSyC7dspQ5tdq3wls3SMD2m8i-Qm7U5HxkBo", "AIzaSyAoL1WfyGYig9J8-cMyuP8gZnvRtqSLptY", "AIzaSyDShvcTEITKYOg-GBc6MrIOcejoGRBnmdM", "AIzaSyBf_4E1RSm_9ABAfUaqW6C6HMNKXoN-_Vs","AIzaSyAmqpWK3R3EsZl-88smB9mlWcCkgXYyeLE", "AIzaSyAxsW6O8-E9_JUT38qKDRj6CwDH3OwJIkE","AIzaSyAabSu4IJvCc1EFsnKJiQPXIHZvQC58_hE","AIzaSyDKElwFyffOhsLY3e7AKZ4ftydGMXroVcc","AIzaSyCyVVfVdpwGB1ucNQDHDvb7V_JhUvhm2IU","AIzaSyBQxLBV979g39IjVGaoNr2ieruMU8nSnG0","AIzaSyDGwbccoNMjlPbgSfyvur7MMk8a1Gf6vwo","AIzaSyAUBViaoZNzIzzvqG8bVNuOqjYDE_p2aGI","AIzaSyDhWvjPMJs7w4g8t-Tj0ty9tgotL3hxqps"]
        if len(remain) != 0:
            main(remain, keys_list[:len(remain)], filename)
