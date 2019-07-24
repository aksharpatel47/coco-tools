import sys
import os
from object_detection.utils import label_map_util, dataset_util
from labels import (LabelJSON, convert_label_dict_to_obj,
                    convert_labels_to_names, get_data_obj_from_xml,
                    get_label_category_dict, hflip_label)
from dataset import ImageDataSet
from PIL import Image
from object_detection.utils.label_map_util import (
    convert_label_map_to_categories, load_labelmap)
from object_detection.utils import dataset_util, label_map_util
import copyreg
import glob
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List
from xml.etree.ElementTree import Comment, Element, SubElement, tostring
import numpy as np
import tensorflow as tf
from coco_image import InferImage


# from .image import Image


def create_tf_record(data, key, encoded_jpg, label_map_dict, flipped=False, ignore_difficult_instances=True):
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    if flipped and data.get("object"):
        for obj in data["object"]:
            hflip_label(width, obj)

    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))

    return example


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       image_types):
    image_subdirectory = ""
    img_path = os.path.join(
        data['folder'], image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)

    examples = []

    vals = []

    if "default" in image_types:
        im = Image.open(full_path)
        img = io.BytesIO()
        im.save(img, format="JPEG")
        encoded_jpg = img.getvalue()
        key = hashlib.sha256(encoded_jpg).hexdigest()
        examples.append(create_tf_record(
            data, key, encoded_jpg, label_map_dict))

    if "flipped" in image_types:
        im = Image.open(full_path)
        fim = im.transpose(Image.FLIP_LEFT_RIGHT)
        img = io.BytesIO()
        fim.save(img, format="JPEG")
        encoded_jpg = img.getvalue()
        data["filename"] = data["filename"].replace(".jpg", "_flipped.jpg")
        key = hashlib.sha256(encoded_jpg).hexdigest()

        examples.append(create_tf_record(
            data, key, encoded_jpg, label_map_dict, flipped=True))

    return examples


def write_tf_record(inputs: List[ImageDataSet], label_path: str, output_file: str):
    writer = tf.python_io.TFRecordWriter(output_file)

    label_dict = label_map_util.get_label_map_dict(label_path)
    label_categories_dict = get_label_category_dict(label_dict)

    label_json = LabelJSON(label_dict)

    for inp in inputs:
        all_input_xmls = glob.glob(os.path.join(
            inp.folder_name, "**", "*.xml"), recursive=True)
        all_input_xmls = list(filter(lambda x: os.path.isfile(
            x.replace(".xml", ".jpg")), all_input_xmls))

        if len(all_input_xmls) > 0:
            for xml_file in all_input_xmls:
                data = get_data_obj_from_xml(xml_file)
                convert_labels_to_names(
                    data, label_dict, label_categories_dict)

                if inp.is_ground_truth:
                    label_json.add_data(data)

                if data.get("object"):
                    difficult_labels = list(filter(lambda x: x == '1', map(
                        lambda x: x["difficult"], data["object"])))

                    if len(difficult_labels) > 0:
                        continue

                try:
                    tf_examples = dict_to_tf_example(
                        data, inp.folder_name, label_dict, inp.image_augmentation)
                    for tfe in tf_examples:
                        writer.write(tfe.SerializeToString())
                except:
                    print(data)
        else:
            all_input_images = glob.glob(os.path.join(
                inp.folder_name, "**", "*.jpg"), recursive=True)

            for input_image in all_input_images:
                im = InferImage(input_image)
                writer.write(im.tfrecord.SerializeToString())

    writer.close()

    if len(label_json.inputs) > 0:
        with open("groundtruth.json", "w") as fd:
            json.dump(label_json.inputs, fd)


def export_tfrecord_to_xmls(tfrecord: str, output_dir: str, label_pbtxt: str, num_categories: int, req_score: float = 0.5):
    label_arr = convert_label_map_to_categories(
        load_labelmap(label_pbtxt), num_categories)
    labels = {}
    json_data = list()

    for val in label_arr:
        labels[val["id"]] = val["name"]

    for example in tf.python_io.tf_record_iterator(tfrecord):
        image_data = dict()
        json_message = tf.train.Example.FromString(example)
        features = json_message.features.feature
        height = features['image/height'].int64_list.value[0]
        width = features['image/width'].int64_list.value[0]
        scores = features['image/detection/score'].float_list.value
        im_labels = features['image/detection/label'].int64_list.value
        xmin = features['image/detection/bbox/xmin'].float_list.value
        xmax = features['image/detection/bbox/xmax'].float_list.value
        ymin = features['image/detection/bbox/ymin'].float_list.value
        ymax = features['image/detection/bbox/ymax'].float_list.value
        file_name = features['image/filename'].bytes_list.value[0].decode(
            "utf-8")
        xannotation = Element("annotation")
        xfilename = SubElement(xannotation, "filename")
        xfilename.text = file_name
        image_data["image_id"] = file_name
        xsize = SubElement(xannotation, "size")
        xwidth = SubElement(xsize, "width")
        xwidth.text = str(width)
        xheight = SubElement(xsize, "height")
        xheight.text = str(height)
        xdepth = SubElement(xsize, "depth")
        xdepth.text = "3"
        xsegmented = SubElement(xannotation, "segmented")
        xsegmented.text = "0"

        boxes = list()
        lbl_scores = list()
        classes = list()

        for i in range(len(im_labels)):
            if scores[i] > req_score:
                xobject = SubElement(xannotation, "object")
                xname = SubElement(xobject, "name")
                xname.text = labels[im_labels[i]]
                classes.append(im_labels[i])
                xscore = SubElement(xobject, "score")
                xscore.text = str(scores[i])
                lbl_scores.append(scores[i])
                xpose = SubElement(xobject, "pose")
                xpose.text = "Unspecified"
                xtruncated = SubElement(xobject, "truncated")
                xtruncated.text = "0"
                xdifficult = SubElement(xobject, "difficult")
                xdifficult.text = "0"
                xbndbox = SubElement(xobject, "bndbox")
                xxmin = SubElement(xbndbox, "xmin")
                xxmin.text = str(round(xmin[i] * width))
                xymin = SubElement(xbndbox, "ymin")
                xymin.text = str(round(ymin[i] * height))
                xxmax = SubElement(xbndbox, "xmax")
                xxmax.text = str(round(xmax[i] * width))
                xymax = SubElement(xbndbox, "ymax")
                xymax.text = str(round(ymax[i] * height))
                boxes.append([round(ymin[i] * height), round(xmin[i] * width),
                              round(xmax[i] * width), round(ymax[i] * height)])

        image_data["detection_boxes"] = boxes
        image_data["detection_scores"] = lbl_scores
        image_data["detection_classes"] = classes

        json_data.append(image_data)

        xstr = tostring(xannotation)

        image_name, _ = os.path.splitext(file_name)
        new_file_path = os.path.join(output_dir, image_name + ".xml")
        with open(new_file_path, "wb") as fd:
            fd.write(xstr)

    # print(json_data)

    with open("detection.json", "w") as fd:
        json.dump(json_data, fd)
