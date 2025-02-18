import json
import os
import csv
from typing import List, Dict, Any

def create_entry(entry, model, method, result_list=None, score=None, evaluation=None):
    """
    Create a new entry from a dataset item with additional metadata.

    Parameters:
        entry (dict): Original dataset entry.
        model (str): The model name used.
        method (str): The jailbreak method or approach applied.
        result_list (list, optional): List of generated responses.
        score (list, optional): List of scores (e.g., from a strongreject evaluator).
        evaluation (list, optional): List of evaluation outputs or reasoning.

    Returns:
        dict: A new dictionary containing selected original fields and additional metadata.
    """
    if result_list is None:
        result_list = []
    if score is None:
        score = []
    if evaluation is None:
        evaluation = []
    
    new_entry = {}
    new_entry["ID"] = entry.get("ID")
    new_entry["Model"] = model
    new_entry["Jailbreak Method"] = method
    new_entry["Full Phrase Phrase"] = entry.get("Full Phrase", "")
    new_entry["Category"] = entry.get("Category", "")
    new_entry["Subcategory"] = entry.get("Subcategory", "")
    new_entry["Prompt"] = entry.get("Prompt", "")
    if result_list:
        new_entry["Responses"] = result_list
    if score:
        new_entry["Score"] = score
    if evaluation:
        new_entry["Evaluation"] = evaluation
    
    return new_entry

def write_results(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
            print(f"Predictions successfully saved to {file_path}")
    except Exception as e:
        print(f"Error writing predictions to file: {e}")

def write_csv(data: List[Dict[str, Any]], file_path: str):
    if not data:
        print(f"No data to write to CSV: {file_path}")
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    columns = set()
    for row in data:
        columns.update(row.keys())
    columns = list(columns)

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"CSV successfully saved to {file_path}")