from .image import Image
import tensorflow as tf
from object_detection.utils import dataset_util
from concurrent.futures import ProcessPoolExecutor
import copyreg
from functools import partial
import os


def write_inference_record(file_name, images):
    output_path = "{}.record".format(file_name)
    writer = tf.python_io.TFRecordWriter(output_path)
    for image in images:
        writer.write(image.tfrecord.SerializeToString())
    writer.close()


def write_record(set_name, images, height, width):
    output_path = "{}.record".format(set_name)
    writer = tf.python_io.TFRecordWriter(output_path)
    crosswalk_tf = partial(create_crosswalk_tf, height=height, width=width)
    with ProcessPoolExecutor() as executor:
        for example in executor.map(crosswalk_tf, images):
            if example and len(example) == 1:
                writer.write(example[0].SerializeToString())
            elif example and len(example) == 2:
                writer.write(example[0].SerializeToString())
                writer.write(example[1].SerializeToString())
    writer.close()


def read_output_record(file_name):
    data = {}

    for example in tf.python_io.tf_record_iterator(file_name):
        json_message = tf.train.Example.FromString(example)
        features = json_message.features.feature
        scores = features['image/detection/score'].float_list.value
        labels = features['image/detection/label'].int64_list.value
        file_name = features['image/filename'].bytes_list.value[0].decode(
            "utf-8")
        image_name, _ = os.path.splitext(file_name)
        detected_labels = set()
        for i, label in enumerate(labels):
            if scores[i] > 0.5:
                detected_labels.add(int(label))
        data[image_name] = list(detected_labels)

    return data


def get_flipped_crosswalk_tf(image, width, height):
    filename = os.path.splitext(image.name)[0] + "_flipped.jpg"
    image_format = b'jpg'

    encoded_jpg = image.get_flipped_image()

    flipped_labels = image.get_flipped_labels()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for lbl_box in flipped_labels:
        xmins.append(float(lbl_box.bb.left) / width)
        ymins.append(float(lbl_box.bb.top) / height)
        xmaxs.append(float(lbl_box.bb.right) / width)
        ymaxs.append(float(lbl_box.bb.bottom) / height)
        classes_text.append(lbl_box.lbl.name.encode("utf8"))
        classes.append(lbl_box.lbl.id)

    filename = image.name.encode("utf8")

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def create_crosswalk_tf(image, height=640, width=480):
    filename = image.name
    image_format = b'jpg'

    try:
        with tf.gfile.GFile(image.path, "rb") as fid:
            encoded_jpg = fid.read()
    except:
        return None

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for lbl_box in image.labels:
        xmins.append(float(lbl_box.bb.left) / width)
        ymins.append(float(lbl_box.bb.top) / height)
        xmaxs.append(float(lbl_box.bb.right) / width)
        ymaxs.append(float(lbl_box.bb.bottom) / height)
        classes_text.append(lbl_box.lbl.name.encode("utf8"))
        classes.append(lbl_box.lbl.id)

    filename = image.name.encode("utf8")

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    if len(image.labels) >= 1:
        return (tf_example, get_flipped_crosswalk_tf(image, width, height))
        # return (tf_example,)
    else:
        return (tf_example,)
