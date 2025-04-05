import json, os

from datasets import load_dataset

import multiprocessing
import time
from glob import glob
from tqdm import tqdm
from google import genai
from google.genai import types
from langchain_core.output_parsers import JsonOutputParser

response_rewriting_prompt = """You are the text splitter and summary tools. Please split the given document into chunks of text that have specific content and generate a title that describes the main subject of each chunk:
[document]: {document}
Please output in JSON format as follows: {{"chunk": [{"title": "title", "content": "text"}]}}."""


def process_item(document, filename):
    parser = JsonOutputParser()
    retries = 3
    attempt = 0

    while attempt < retries:
        try:

            client = genai.Client(
                api_key="",
            )
            
            response = client.models.generate_content(
                model='gemini-2.0-flash', contents=response_rewriting_prompt.format(document=document), 
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=200,
                    ),
                )
            chunks = parser.parse(response.text)["chunk"]
            for chunk in chunks:
                new_item = {}
                new_item["title"] = chunk["title"]

                new_item["document_id"] = filename.split(".txt")[0]
                new_item["document"] = chunk["content"]
            with open(f"documents.json", "a") as f:
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            return new_item, True

        except Exception as e:
            print(f"Error processing item {filename}  attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)  
            else:
                print(f"Failed after {retries} attempts.")
                return filename, False 
            
def process_items_parallel(items: list, num_processes: int = None) -> list:
    """Processes items in parallel using multiprocessing."""

    if num_processes is None:
      num_processes = multiprocessing.cpu_count() 
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_item, [(item[0], item[1]) for item in items])

    return results


if __name__ == "__main__":
    list_file = [os.path.basename(i) for i in glob("vinfast_docs_en/*.txt")]
    documents = []
    for filename in tqdm(list_file):
        f = open(f"vinfast_docs_en/{filename}", "r")
        documents.append([f.read(), filename])
    process_items_parallel(documents, 10)
