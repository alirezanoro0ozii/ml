import os
import json
import matplotlib.pyplot as plt
from collections import Counter

def read_dmp_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            fields = [field.strip() for field in line.split('|')]
            if fields[-1] == '':
                fields = fields[:-1]
            results.append(fields)
    return results

def read_names_dmp(file_path):
    names_dict = {}
    for fields in read_dmp_file(file_path):
        tax_id = int(fields[0])
        name = fields[1]
        name_type = fields[3]
        if name_type == "scientific name":
            names_dict[tax_id] = name
    return names_dict

if not os.path.exists('taxon/names.json'):
    names_dict = read_names_dmp('taxon/names.dmp')
    with open('taxon/names.json', 'w') as f:
        json.dump(names_dict, f)
else:
    with open('taxon/names.json', 'r') as f:
        names_dict = json.load(f)


def parse_nodes_dmp(file_path):
    taxonomy_tree = {}
    for fields in read_dmp_file(file_path):
        tax_id = int(fields[0])
        parent_tax_id = int(fields[1])
        # print("tax_id: ", tax_id, "parent_tax_id: ", parent_tax_id)
        taxonomy_tree[tax_id] = parent_tax_id
    return taxonomy_tree


if not os.path.exists('taxon/nodes.json'):
    nodes = parse_nodes_dmp('taxon/nodes.dmp')
    with open('taxon/nodes.json', 'w') as f:
        json.dump(nodes, f) 
else:
    with open('taxon/nodes.json', 'r') as f:
        nodes = json.load(f)


def parse_ranks_dmp(file_path):
    ranks = {}
    for fields in read_dmp_file(file_path):
        tax_id = int(fields[0])
        rank = fields[2]
        # print("tax_id: ", tax_id, "rank: ", rank)
        ranks[tax_id] = rank
    return ranks

if not os.path.exists('taxon/ranks.json'):
    ranks = parse_ranks_dmp('taxon/nodes.dmp')
    with open('taxon/ranks.json', 'w') as f:
        json.dump(ranks, f)
else:
    with open('taxon/ranks.json', 'r') as f:
        ranks = json.load(f)

if not os.path.exists('taxon/parent_to_children.json'):
    # Create a dictionary where keys are parent_tax_ids and values are lists of child tax_ids
    parent_to_children = {}

    # Iterate through the nodes dictionary where key is tax_id and value is parent_tax_id
    for tax_id, parent_tax_id in nodes.items():
        tax_id = int(tax_id)
        parent_tax_id = int(parent_tax_id)

        # Add the tax_id to the list of children for this parent_tax_id
        if parent_tax_id not in parent_to_children:
            parent_to_children[parent_tax_id] = []
        parent_to_children[parent_tax_id].append(tax_id)

    with open('taxon/parent_to_children.json', 'w') as f:
        json.dump(parent_to_children, f)
else:
    with open('taxon/parent_to_children.json', 'r') as f:
        parent_to_children = json.load(f)

def trace_back(first_one, show=True):
    path = [first_one]
    while 1:
        # print(nodes[first_one], " --> ", first_one)
        path.append(nodes[first_one])
        if nodes[first_one] == 1:
            break
        first_one = str(nodes[first_one])
    if show:
        for p in path[::-1]:
            print("Children of tax ID {:<8} {:<30} {:<15} {}".format(
                p,
                names_dict[str(p)],
                ranks[str(p)],
                parent_to_children.get(str(p), [])
            ))

    return path

