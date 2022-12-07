import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
import pickle
import csv

from pycocotools.coco import COCO

coco_annotation_file_path = "coco/annotations/instances_train2017.json"
coco_annotation = COCO(annotation_file=coco_annotation_file_path)

generate_ratio = False

def main():
    cooccurence_table = dict()

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.    

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    for cat_name in cat_names:
        cooccurence_table[cat_name] = dict()
        for cat_name2 in cat_names:
            cooccurence_table[cat_name][cat_name2] = 0
        cooccurence_table[cat_name]['end node'] = 0

    for cat_id in cat_ids:
        img_ids = coco_annotation.getImgIds(catIds=[cat_id])

        # Get all the annotations for the specified image.
        for img_id in img_ids:
            cats_in_img = []
            ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)

            anns = coco_annotation.loadAnns(ann_ids)
            for ann in anns:
                cats_in_img.append(ann['category_id'])
            cats_in_img = set(cats_in_img)

            instance = 0
            for cat_in_img in cats_in_img:
                if cat_in_img == cat_id:
                    continue
                # if id_to_name(cat_in_img) not in cooccurence_table[id_to_name(cat_id)]:
                #     cooccurence_table[id_to_name(cat_id)][id_to_name(cat_in_img)] = 0
                cooccurence_table[id_to_name(cat_id)][id_to_name(cat_in_img)] += 1
                instance += 1

            if instance == 0:
                cooccurence_table[id_to_name(cat_id)]['end node'] += 1

    if generate_ratio:
        for cat_id in cat_ids:
            total = sum(cooccurence_table[id_to_name(cat_id)].values(), 0.0)
            cooccurence_table[id_to_name(cat_id)] = {k: v / total for k, v in cooccurence_table[id_to_name(cat_id)].items()}

    # print_cooccurence_table(cooccurence_table)

    # Save the table into pkl
    with open('cooccurence_table.pkl', 'wb') as f:
        pickle.dump(cooccurence_table, f)
    
    # Save the table into csv format
    save_to_csv(cooccurence_table)


    return

def id_to_name(query_id):
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    return query_name

def name_to_id(query_name):
    query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
    return query_id

def save_to_csv(table):
    csv_columns = [key for key in table]
    csv_columns.insert(0, ' ')
    csv_columns.append('end node')

    if generate_ratio:
        filename = 'cooccurence_ratio_table.csv'
    else:
        filename = 'cooccurence_table.csv'

    with open(filename, 'w') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(csv_columns)
        for key, value in table.items():
            data = []
            data.append(key)
            for k, v in value.items():
                data.append(round(v,4))
            wr.writerow(data)
    return

def print_cooccurence_table(table):
    for key, value in table.items():
        print(key, ":")
        for k, v in sorted(value.items(), key=lambda item: item[1], reverse=True):
            print("   ", k, ":", round(v,3))
    return


if __name__ == "__main__":

    main()