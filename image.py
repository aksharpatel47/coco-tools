# from labels import Label, LabelBox, BoundingBox
# import xml.etree.ElementTree as ET
# from xml.etree.ElementTree import Element, SubElement, tostring
import os
from PIL import Image as IM
import tensorflow as tf
from object_detection.utils import dataset_util
import io
from enum import Enum
from typing import List

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