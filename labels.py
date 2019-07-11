from collections import namedtuple
import os
from typing import Optional, List
from lxml import etree
from object_detection.utils import dataset_util

BoundingBox = namedtuple('BoundingBox', ['left', 'top', 'right', 'bottom'])


def get_label_category_dict(label_dict):
    scategory = {}
    for k in label_dict.keys():
        scategory["".join(map(lambda x: x[0], k.split("-")))] = k

    return scategory


def hflip_label(image_width, label):
    xmax = int(image_width) - int(label["bndbox"]["xmin"])
    xmin = int(image_width) - int(label["bndbox"]["xmax"])

    if xmax < 0 or xmin < 0:
        raise Exception("Zero", xmax, xmin)

    if xmax > image_width or xmin > image_width:
        raise Exception("Out of Bounds", xmax, xmin)

    if xmin >= xmax:
        raise Exception("Comparison", xmin, xmax)

    label["bndbox"]["xmin"] = str(xmin)
    label["bndbox"]["xmax"] = str(xmax)


def convert_label_dict_to_obj(data, label_map_dict, flipped=False):
    boxes = []
    scores = []
    classes = []

    if 'object' in data:
        for obj in data['object']:
            if bool(int(obj['difficult'])):
                continue

            scores.append(1.0)
            classes.append(label_map_dict[obj['name']])
            boxes.append([float(obj['bndbox']['ymin']), float(obj['bndbox']['xmin']), float(
                obj['bndbox']['xmax']), float(obj['bndbox']['ymax'])])

    filename = data["filename"].replace(
        ".jpg", "_flipped.jpg") if flipped else data["filename"]

    return {"image_id": filename, "groundtruth_boxes": boxes, "groundtruth_scores": scores, "groundtruth_classes": classes}


def get_data_obj_from_xml(path_str):
    with open(path_str) as fd:
        xml_str = fd.read()
        xml_str = xml_str.replace("<object></object>", "")

    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)["annotation"]
    data["folder"] = os.path.basename(os.path.split(path_str)[0])

    return data


def convert_labels_to_names(data, label_dict, label_categories_dict):
    labels_to_skip = []
    if data.get("object"):
        if len(data["object"]) > 0:
            for i, obj in enumerate(data["object"]):
                if len(obj["name"]) <= 2:
                    if label_categories_dict.get(obj["name"]):
                        data["object"][i]["name"] = label_categories_dict[obj["name"]]
                    else:
                        labels_to_skip.append(obj)
                elif not label_dict.get(obj["name"]):
                    labels_to_skip.append(obj)

            for lbl in labels_to_skip:
                data["object"].remove(lbl)
        else:
            del data["object"]


class LabelJSON():
    def __init__(self, label_dict: dict):
        self.inputs = []
        self.label_dict = label_dict

    def add_data(self, data_obj, flipped=False):
        self.inputs.append(convert_label_dict_to_obj(
            data_obj, self.label_dict, flipped=flipped))
