from collections import namedtuple
import os
from typing import Optional

BoundingBox = namedtuple('BoundingBox', ['left', 'top', 'right', 'bottom'])


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
