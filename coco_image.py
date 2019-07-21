# from labels import Label, LabelBox, BoundingBox
# import xml.etree.ElementTree as ET
# from xml.etree.ElementTree import Element, SubElement, tostring
import os
from PIL import Image as IM
from PIL import ImageOps
import tensorflow as tf
from object_detection.utils import dataset_util
import io
from enum import Enum
from typing import List
from labels import get_data_obj_from_xml
from xml.etree.ElementTree import Comment, Element, SubElement, tostring


class ImageAugmentation(Enum):
    DEFAULT = 1
    FLIPPED = 2


class InferImage:
    def __init__(self, path):
        self.path = path
        self.im = IM.open(self.path)
        self.img = self.get_image()
        self.tfrecord = self.get_tf_record()

    def get_image(self):
        img = io.BytesIO()
        self.im.save(img, format='JPEG')
        return img.getvalue()

    def get_image_size(self):
        width, height = self.im.size
        return width, height

    def get_tf_record(self):
        image_format = b'jpg'
        width, height = self.get_image_size()
        _, image_name = os.path.split(self.path)

        image_name = image_name.encode("utf8")
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(image_name),
            'image/source_id': dataset_util.bytes_feature(image_name),
            'image/encoded': dataset_util.bytes_feature(self.img),
            'image/format': dataset_util.bytes_feature(image_format)
        }))

        return tf_example


def _export_data_to_xml(data, path: str):
    xannotation = Element("annotation")
    xfilename = SubElement(xannotation, "filename")
    xfilename.text = data["filename"]
    xfolder = SubElement(xannotation, "folder")
    xfolder.text = data["folder"]
    if data.get("path"):
        xpath = SubElement(xannotation, "path")
        xpath.text = data["path"]
    xsize = SubElement(xannotation, "size")
    xwidth = SubElement(xsize, "width")
    xwidth.text = data["size"]["width"]
    xheight = SubElement(xsize, "height")
    xheight.text = data["size"]["height"]
    xdepth = SubElement(xsize, "depth")
    xdepth.text = "3"
    xsegmented = SubElement(xannotation, "segmented")
    xsegmented.text = "0"
    xsource = SubElement(xannotation, "source")
    xdatabase = SubElement(xsource, "database")
    xdatabase.text = "Unknown"

    if data.get("object") and len(data["object"]) > 0:
        for obj in data["object"]:
            xobject = SubElement(xannotation, "object")
            xname = SubElement(xobject, "name")
            xname.text = obj["name"]
            if obj.get("score"):
                xscore = SubElement(xobject, "score")
                xscore.text = obj["score"]
            xpose = SubElement(xobject, "pose")
            xpose.text = "Unspecified"
            xtruncated = SubElement(xobject, "truncated")
            xtruncated.text = "0"
            xdifficult = SubElement(xobject, "difficult")
            xdifficult.text = obj["difficult"]
            xbndbox = SubElement(xobject, "bndbox")
            bndbox = obj["bndbox"]
            xxmin = SubElement(xbndbox, "xmin")
            xxmin.text = bndbox["xmin"]
            xymin = SubElement(xbndbox, "ymin")
            xymin.text = bndbox["ymin"]
            xxmax = SubElement(xbndbox, "xmax")
            xxmax.text = bndbox["xmax"]
            xymax = SubElement(xbndbox, "ymax")
            xymax.text = bndbox["ymax"]

    xstr = tostring(xannotation)

    with open(path, "wb") as fd:
        fd.write(xstr)


class Image:
    def __init__(self, path: str):
        self.path = path
        self.im: IM = IM.open(self.path)
        self.xml_path = self.path.replace(".jpg", ".xml")
        self.data = get_data_obj_from_xml(self.xml_path)

    def make_square(self):
        width, height = self.im.size
        max_size = max(width, height)

        dw = max_size - width
        dh = max_size - height
        tw = dw//2
        th = dh//2
        border = (tw, th)

        self.im = ImageOps.expand(self.im, border)

        if self.data.get("object") and len(self.data["object"]) > 0:
            for obj in self.data["object"]:
                bndbox = obj["bndbox"]
                bndbox["xmin"] = str(int(bndbox["xmin"]) + tw)
                bndbox["xmax"] = str(int(bndbox["xmax"]) + tw)
                bndbox["ymin"] = str(int(bndbox["ymin"]) + th)
                bndbox["ymax"] = str(int(bndbox["ymax"]) + th)

    def resize_image(self, target_width: int, target_height: int):
        cur_width, cur_height = self.im.size
        self.im = self.im.resize(
            (target_width, target_height), resample=IM.BILINEAR)

        if self.data.get("object") and len(self.data["object"]) > 0:
            for obj in self.data["object"]:
                bndbox = obj["bndbox"]
                bndbox["xmin"] = str(int(int(bndbox["xmin"]) *
                                         (target_width / cur_width)))
                bndbox["xmax"] = str(int(int(bndbox["xmax"]) *
                                         (target_width / cur_width)))
                bndbox["ymin"] = str(int(int(bndbox["ymin"]) *
                                         (target_height / cur_height)))
                bndbox["ymax"] = str(int(int(bndbox["ymax"]) *
                                         (target_height / cur_height)))

    def export_to_xml(self):
        _export_data_to_xml(self.data, self.xml_path)

    def save_image(self):
        self.im.save(self.path)
