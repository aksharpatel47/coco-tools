from collections import namedtuple
import os
from typing import Optional
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


def convert_label_dict_to_obj(data, label_map_dict):
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

    return {"image_id": data["filename"], "groundtruth_boxes": boxes, "groundtruth_scores": scores, "groundtruth_classes": classes}


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


def get_bb_size(bb):
    width = bb.right - bb.left
    height = bb.bottom - bb.top
    return width * height


class Label:
    def __init__(self, id, name):
        self.id = id
        self.name = name


class LabelBox:
    def __init__(self, lbl, bb: BoundingBox, difficult: bool, tag: str = None):
        self.lbl = lbl
        self.bb = bb
        self.bb_size = get_bb_size(bb)
        self.difficult = difficult
        self.tag = tag

    def get_flipped_label(self, width, height):
        left = width - self.bb.right
        right = width - self.bb.left
        top = height - self.bb.bottom
        bottom = height - self.bb.top

        return LabelBox(self.lbl, BoundingBox(left, top, right, bottom), self.difficult)

    def get_iou(self, other_label):
        uleft = min(self.bb.left, other_label.bb.left)
        uright = max(self.bb.right, other_label.bb.right)
        utop = min(self.bb.top, other_label.bb.top)
        ubottom = max(self.bb.bottom, other_label.bb.bottom)
        ul = uright - uleft
        ub = ubottom - utop

        ileft = max(self.bb.left, other_label.bb.left)
        iright = min(self.bb.right, other_label.bb.right)
        itop = max(self.bb.top, other_label.bb.top)
        ibottom = min(self.bb.bottom, other_label.bb.bottom)
        il = max(0, iright - ileft)
        ib = max(0, ibottom - itop)

        return (il * ib) / (ul * ub)

    def bigger_if_iou(self, other_label):
        self_size = ((self.bb.right - self.bb.left)
                     * (self.bb.bottom - self.bb.top))
        other_size = ((other_label.bb.right - other_label.bb.left)
                      * (other_label.bb.bottom - other_label.bb.top))

        if self_size > other_size:
            return other_label
        else:
            return self

    def is_superset(self, other_label) -> bool:
        uleft = min(self.bb.left, other_label.bb.left)
        uright = max(self.bb.right, other_label.bb.right)
        utop = min(self.bb.top, other_label.bb.top)
        ubottom = max(self.bb.bottom, other_label.bb.bottom)
        ul = uright - uleft
        ub = ubottom - utop

        cl = self.bb.right - self.bb.left
        cb = self.bb.bottom - self.bb.top

        return (ul * ub) / (cl * cb) <= 1.2


label_initials = ["z", "l", "c"]

label_names = {
    "z": "zcrosswalk",
    "l": "lcrosswalk",
    "c": "curbcut"
}


def get_label_dict(label_initials, label_names):
    label_dict = {}
    for i, lbl in enumerate(label_initials):
        label_dict[lbl] = Label(i + 1, label_names[lbl])

    return label_dict


def get_base_data(images):
    base_data = {}

    for im in images:
        file_name, extension = os.path.splitext(im.name)
        print(im.labels)
        base_data[file_name] = list(
            map(lambda x: x.lbl.id, im.labels))

    return base_data
