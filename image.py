from tools.labels import Label, LabelBox, BoundingBox
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, tostring
import os
from PIL import Image as IM
import tensorflow as tf
from object_detection.utils import dataset_util
import io


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

    def get_tf_record(self):
        image_format = b'jpg'
        width, height = self.im.size
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


class ImageLabel:

    def __init__(self, path: str):
        self.path = path
        self.labels = []
        self.read_xml()

    def read_xml(self):
        tree = ET.parse(self.path)
        root = tree.getroot()

        for child in root:
            if child.tag == "filename":
                self.image_name: str = child.text
            elif child.tag == "size":
                self.read_image_info(child)
            elif child.tag == "object":
                self.read_label_info(child)

    def read_label_info(self, child):
        label_info = {
            "bndbox": {}
        }

        for l_c in child:
            if l_c.tag == "name":
                label_info["name"] = l_c.text
            elif l_c.tag == "difficult":
                label_info["difficult"] = l_c.text
            elif l_c.tag == "bndbox":
                for bc in l_c:
                    if bc.tag == "xmin":
                        label_info["bndbox"]["xmin"] = bc.text
                    elif bc.tag == "xmax":
                        label_info["bndbox"]["xmax"] = bc.text
                    elif bc.tag == "ymin":
                        label_info["bndbox"]["ymin"] = bc.text
                    elif bc.tag == "ymax":
                        label_info["bndbox"]["ymax"] = bc.text

        self.labels.append(label_info)

    def merge_new_labels(self, new_labels):
        self.labels.extend(new_labels)

    def read_image_info(self, child):
        for size_c in child:
            if size_c.tag == "width":
                self.image_width = int(size_c.text)
            elif size_c.tag == "height":
                self.image_height = int(size_c.text)
            elif size_c.tag == "depth":
                self.image_depth = int(size_c.text)

    def write_xml(self, output_dir: str):
        output_path = os.path.join(
            output_dir, self.image_name.replace(".jpg", ".xml"))

        xannotation = Element("annotation")
        xfilename = SubElement(xannotation, "filename")
        xfilename.text = self.image_name.replace(".xml", ".jpg")
        xsize = SubElement(xannotation, "size")
        xwidth = SubElement(xsize, "width")
        xwidth.text = str(self.image_width)
        xheight = SubElement(xsize, "height")
        xheight.text = str(self.image_height)
        xdepth = SubElement(xsize, "depth")
        xdepth.text = "3"
        xsegmented = SubElement(xannotation, "segmented")
        xsegmented.text = "0"

        for lbl in self.labels:
            xobject = SubElement(xannotation, "object")
            xname = SubElement(xobject, "name")
            xname.text = lbl["name"]
            xpose = SubElement(xobject, "pose")
            xpose.text = "Unspecified"
            xtruncated = SubElement(xobject, "truncated")
            xtruncated.text = "0"
            xdifficult = SubElement(xobject, "difficult")
            xdifficult.text = lbl["difficult"]
            xbndbox = SubElement(xobject, "bndbox")
            xxmin = SubElement(xbndbox, "xmin")
            xxmin.text = str(lbl["bndbox"]["xmin"])
            xymin = SubElement(xbndbox, "ymin")
            xymin.text = str(lbl["bndbox"]["ymin"])
            xxmax = SubElement(xbndbox, "xmax")
            xxmax.text = str(lbl["bndbox"]["xmax"])
            xymax = SubElement(xbndbox, "ymax")
            xymax.text = str(lbl["bndbox"]["ymax"])

        with open(output_path, "wb") as fd:
            fd.write(tostring(xannotation))


class Image:

    def __init__(self, path, label_dict):
        self.path = path
        self.pathc = os.path.split(self.path)
        self.name = self.pathc[len(self.pathc) - 1]
        self.labels = []
        self.label_tags = []
        self.label_dict = label_dict
        self.width = 0
        self.height = 0
        self.folder_name = ""

    def add_label(self, label):
        self.labels.append(label)

    def load_labels(self):
        xml_path = self.path.replace(".jpg", ".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for child in root:
            if child.tag == "object":
                left = 0
                top = 0
                bottom = 0
                right = 0
                difficult = False
                lbl_name = ""

                for obj_child in child:
                    if obj_child.tag == "name":
                        lbl_name = obj_child.text
                    elif obj_child.tag == "difficult":
                        difficult = bool(int(obj_child.text))
                    elif obj_child.tag == "bndbox":
                        for bc in obj_child:
                            if bc.tag == "xmin":
                                left = int(bc.text)
                            elif bc.tag == "ymin":
                                top = int(bc.text)
                            elif bc.tag == "xmax":
                                right = int(bc.text)
                            elif bc.tag == "ymax":
                                bottom = int(bc.text)

                if lbl_name == "d":
                    continue
                if len(lbl_name) > 2:
                    lbl_name = "".join(
                        map(lambda x: x[0], lbl_name.split("-")))
                if self.label_dict.get(lbl_name) is None:
                    continue
                bb = BoundingBox(left, top, right, bottom)
                lbl = self.label_dict[lbl_name]
                lbl_box = LabelBox(lbl, bb, difficult, tag=lbl_name)
                self.label_tags.append(lbl_name)
                self.add_label(lbl_box)

            if child.tag == "size":
                for obj_child in child:
                    if obj_child.tag == "width":
                        self.width = int(obj_child.text)
                    elif obj_child.tag == "height":
                        self.height = int(obj_child.text)

            if child.tag == "folder":
                self.folder_name = child.text

    def get_base_annotation(self) -> Element:
        xannotation = Element("annotation")
        xfilename = SubElement(xannotation, "filename")
        xfilename.text = self.name.replace(".xml", ".jpg")
        xsize = SubElement(xannotation, "size")
        xwidth = SubElement(xsize, "width")
        xwidth.text = str(self.width)
        xheight = SubElement(xsize, "height")
        xheight.text = str(self.height)
        xdepth = SubElement(xsize, "depth")
        xdepth.text = "3"
        xsegmented = SubElement(xannotation, "segmented")
        xsegmented.text = "0"

        return xannotation

    def set_object_tags(self, xannotation):
        for i in range(len(self.labels)):
            lbl = self.labels[i]
            xobject = SubElement(xannotation, "object")
            xname = SubElement(xobject, "name")
            xname.text = lbl.tag
            xpose = SubElement(xobject, "pose")
            xpose.text = "Unspecified"
            xtruncated = SubElement(xobject, "truncated")
            xtruncated.text = "0"
            xdifficult = SubElement(xobject, "difficult")
            xdifficult.text = "0"
            xbndbox = SubElement(xobject, "bndbox")
            xxmin = SubElement(xbndbox, "xmin")
            xxmin.text = str(lbl.bb.left)
            xymin = SubElement(xbndbox, "ymin")
            xymin.text = str(lbl.bb.top)
            xxmax = SubElement(xbndbox, "xmax")
            xxmax.text = str(lbl.bb.right)
            xymax = SubElement(xbndbox, "ymax")
            xymax.text = str(lbl.bb.bottom)

    def get_flipped_image(self):
        im = IM.open(self.path)
        fim = im.transpose(IM.FLIP_LEFT_RIGHT)
        img = io.BytesIO()
        fim.save(img, format="JPEG")
        return img.getvalue()

    def get_flipped_labels(self):
        return [l.get_flipped_label(self.width, self.height) for l in self.labels]


if __name__ == '__main__':
    import os
    import glob

    image_list = glob.glob(os.path.join(
        'C:\\Users\\aksha\\Downloads\\DupontImages\\Akshar', "*.jpg"))
    images = [InferImage(p) for p in image_list]
