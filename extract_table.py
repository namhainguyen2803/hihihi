import re
import json

def extract_values(table_content):
    method_values = {}
    lines = table_content.split('\n')
    for line in lines:
        if line.strip() == "":
            continue
        parts = line.split('&')
        print(parts)
        method_name = parts[0].strip()
        values = [float(part.strip().split()[0].replace('$', '')) for part in parts[1:]]
        method_values[method_name] = parts[1:]
    return method_values

def read_whole_table_from_file(file_path):
    with open(file_path, 'r') as file:
        whole_table = file.read()
    return whole_table

def extract_table_content(whole_table):
    rows = whole_table.split('\n')

    table_content = []
    for row in rows:
        if any(method in row for method in ['USWB', 'MFSWB']):
            table_content.append(row.strip())
    return '\n'.join(table_content)

def group_values_by_kappa(dict_values):
    grouped_dict = {'fsw_0.1': {}, 'fsw_0.5': {}, 'fsw_1.0': {}, 'fsw_2.0': {}, 'fsw_4.0': {}}
    for key, values in dict_values.items():
        kappa_values = [values[i:i + 2] for i in range(0, len(values), 2)]
        for i, (kappa_key, value) in enumerate(zip(grouped_dict.keys(), kappa_values)):
            if key not in grouped_dict[kappa_key]:
                grouped_dict[kappa_key][key] = {'F': value[0], 'W': value[1]}
            else:
                grouped_dict[kappa_key][key] = {'F': value[0], 'W': value[1]}
    return grouped_dict

def extract_caption(table_content):
    caption_start = table_content.find("\\caption{")
    if caption_start != -1:
        caption_start += len("\\caption{")
        caption_end = table_content.find("}", caption_start)
        if caption_end != -1:
            return table_content[caption_start:caption_end]
    return None

def extract_info(caption_line):
    space_index = caption_line.find("space")
    epochs_index = caption_line.find("epochs")
    
    if space_index != -1 and epochs_index != -1:
        space_word = caption_line[:space_index].split()[-1]
        epochs_number = caption_line[:epochs_index].split()[-1]
        
        return space_word, epochs_number
    else:
        return None, None

if __name__ == "__main__":
    file_path = "table_fairness.txt"

    with open(file_path, "r") as file:
        text = file.read()

    tables = text.split(r"\end{table}")
    
    big_dick = dict()

    for i, table in enumerate(tables):
        caption_line = extract_caption(table)
        if caption_line:
            space, epochs = extract_info(caption_line)
            print("Space:", space)
            print("Epochs:", epochs)
            
            table_content = extract_table_content(table)
            dict_values = extract_values(table_content)
            if space not in big_dick:
                big_dick[space] = {}
            
            # Assign the values to the nested dictionary
            big_dick[space][epochs] = group_values_by_kappa(dict_values)

    # print(big_dick)

    json_file_path = "big_dick_data.json"

    with open(json_file_path, "w") as json_file:
        json.dump(big_dick, json_file, indent=4)

    print("Dictionary saved to JSON file:", json_file_path)
