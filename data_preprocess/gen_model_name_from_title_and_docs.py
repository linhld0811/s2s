import json

from datasets import load_dataset

import multiprocessing
import time
from google import genai
from google.genai import types
from langchain_core.output_parsers import JsonOutputParser

prompt = """you are an model name extracter. Given title and documents name, please extracter name of model motor:
- Use the title with a higher priority than the document name
- Return with json format: {{"model_name": ["model_name"]}}
Example:
Title: Traction Mode: Enabling and Function
Document name: VF9_Owner_Manual_071124_5
Return: {{"model_name": ["VF9"]}}
Title: VF8S LUX and LUX PLUS Dimensions and Capacities
Document name: Brochure_VF8_Lux_161224_0
Return: {{"model_name": ["VF8S Lux", "VF8S Lux Plus"]}}

Title: {title}
Document name: {document_id}
"""
def process_item(item, filename):
    parser = JsonOutputParser()
    retries = 3
    attempt = 0

    while attempt < retries:
        try:

            client = genai.Client(
                api_key="",
            )
            
            response = client.models.generate_content(
                model='gemini-2.0-flash', contents=prompt.format(title=item["title"], document_id=item["document_id"]), 
                    config=types.GenerateContentConfig(
                        temperature=0,
                    ),
                )
        
            item["model_name"] = parser.parse(response.text)["model_name"]
            with open(filename, "a") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            time.sleep(1)
            print(f"Processing item: {item} with {filename}, attempt {attempt + 1}")
            return item, True

        except Exception as e:
            print(f"Error processing item {item}  attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)  
            else:
                print(f"Failed after {retries} attempts.")
                return item, False
            
def process_items_parallel(items: list, filename: str, num_processes: int = None) -> list:
    """Processes items in parallel using multiprocessing."""

    if num_processes is None:
      num_processes = multiprocessing.cpu_count() # default to cpu count.

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_item, [(item, filename) for item in items])

    return results


if __name__ == "__main__":
    data = load_dataset("data", data_files="documents_chunked.json")["train"]
    dataset = [i for i in data]

    done = []
    try:
        c = open(f"documents_chunked_w_model.json", "r")
        for line in c.readlines():
            done.append(json.loads(line.strip())["title"])
        c.close()
    except:
        pass

    remain = [i for i in dataset if i["title"] not in done]
    print(len(remain), len(dataset))
    if len(remain) != 0:
        process_items_parallel(remain, "documents_chunked_w_model.json", 50)
