# import json
#
# # Opening JSON file
# f = open('local/gummy_smile_project/gum/termal/termal_label.json')
#
# # returns JSON object as
# # a dictionary
# data = json.load(f)
#
# for json_obj in data:
#     filename = json_obj['_id'] + '.json'
#
#     with open(filename, 'w') as out_json_file:
#         # Save each obj to their respective filepath
#         # with pretty formatting thanks to `indent=4`
#         json.dump(json_obj, out_json_file, indent=4)


import os
import json
from itertools import islice

def split_json(
    data_path,
    file_name,
    size_split=1000
):
    """Split a big JSON file into chunks.
    data_path : str, "data_folder"
    file_name : str, "data_file" (exclude ".json")
    """
    with open(os.path.join(data_path, file_name + ".json"), "r") as f:
        whole_file = json.load(f)

    split = len(whole_file["images"]) # size_split

    for i in range(split + 1):
        with open(os.path.join(data_path, file_name + "_"+ str(split+1) + "_" + str(i+1) + ".json"), 'w') as f:
            json.dump(dict(islice(whole_file.items(), i*size_split, (i+1)*size_split)), f)
    return

A = split_json("local/gummy_smile_project/gum/termal","termal_label", 1000 )


import os
import json
from itertools import islice

data_path = "local/gummy_smile_project/gum/termal"

file_name = "4_label_grubu_0-1-2-3"

with open(os.path.join(data_path, file_name + ".json"), "r") as f:
    whole_file = json.load(f)
size_split = 5
split = len(whole_file["images"])  # size_split

for i in range(split + 1):
    with open(os.path.join(data_path, file_name + "_" + str(split + 1) + "_" + str(i + 1) + ".json"), 'w') as f:
        json.dump(dict(islice(whole_file.items(), i * size_split, (i + 1) * size_split)), f)