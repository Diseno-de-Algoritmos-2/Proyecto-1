import json


def save_simulation_to_json(data, filename):
    with open(f"Comparación/results/{filename}.json", "w") as json_file:
        json.dump(data, json_file)
