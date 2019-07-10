import copyreg
import glob
import hashlib
import io
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List

import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util
from PIL import Image

from dataset import ImageDataSet
# from .image import Image
from labels import (convert_label_dict_to_obj, convert_labels_to_names,
                    get_data_obj_from_xml, get_label_category_dict,
                    hflip_label)


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

    for inp in inputs:
        all_input_xmls = glob.glob(os.path.join(
            inp.folder_name, "**", "*.xml"), recursive=True)
        all_input_xmls = list(filter(lambda x: os.path.isfile(
            x.replace(".xml", ".jpg")), all_input_xmls))

        for xml_file in all_input_xmls:
            data = get_data_obj_from_xml(xml_file)
            convert_labels_to_names(data, label_dict, label_categories_dict)
            tf_examples = dict_to_tf_example(
                data, inp.folder_name, label_dict, inp.image_augmentation)

            for tfe in tf_examples:
                writer.write(tfe.SerializeToString())

    writer.close()
