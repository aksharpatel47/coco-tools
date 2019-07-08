import os
from .labels import hflip_label, convert_label_dict_to_obj, convert_labels_to_names, get_data_obj_from_xml, get_label_category_dict
from object_detection.utils import label_map_util

label_path = os.path.join("sample_files", "labels.pbtxt")
label_dict = label_map_util.get_label_map_dict(label_path)
label_categories_dict = get_label_category_dict(label_dict)


def test_get_label_category_dict():
    label_category_dict = get_label_category_dict(label_dict)

    groundtruth_category_dict = {
        "w": "walksignal",
        "wa": "walksignal-alone"
    }

    assert label_category_dict == groundtruth_category_dict


def test_hflip_label():
    lbl = {
        "bndbox": {
            "xmin": 0,
            "xmax": 640
        }
    }

    img_width = 640

    hflip_label(img_width, lbl)

    assert lbl["bndbox"]["xmin"] == '0'
    assert lbl["bndbox"]["xmax"] == '640'

    lbl = {
        "bndbox": {
            "xmin": 5,
            "xmax": 620
        }
    }

    hflip_label(img_width, lbl)

    assert lbl["bndbox"]["xmin"] == '20'
    assert lbl["bndbox"]["xmax"] == '635'


def test_get_data_obj_from_xml():
    no_obj_path_str = os.path.join("sample_files", "no_objects.xml")
    data = get_data_obj_from_xml(no_obj_path_str)

    groundtruth_data = {
        "folder": "sample_files",
        "filename": "33.6491385170296_-112.20791456542071_0_0.jpg",
        "source": {
            "database": "Unknown"
        },
        "size": {
            "width": "512",
            "height": "512",
            "depth": "3"
        },
        "segmented": "0"
    }

    assert data == groundtruth_data

    with_labels_path_str = os.path.join("sample_files", "with_labels.xml")
    data = get_data_obj_from_xml(with_labels_path_str)

    groundtruth_data = {
        "folder": "sample_files",
        "filename": "33.6693057958454_-112.12601933960218_270_0.jpg",
        "source": {
            "database": "Unknown"
        },
        "size": {
            "width": "512",
            "height": "512",
            "depth": "3"
        },
        "segmented": "0",
        "object": [
            {
                "name": "w",
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {
                    "xmin": "85",
                    "xmax": "102",
                    "ymin": "207",
                    "ymax": "270"
                }
            },
            {
                "name": "w",
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {
                    "xmin": "397",
                    "xmax": "426",
                    "ymin": "227",
                    "ymax": "272"
                }
            }
        ]
    }

    assert data == groundtruth_data


def test_convert_labels_to_names():
    with_labels_path_str = os.path.join("sample_files", "with_labels.xml")
    data = get_data_obj_from_xml(with_labels_path_str)
    convert_labels_to_names(data, label_dict, label_categories_dict)

    groundtruth_data = {
        "folder": "sample_files",
        "filename": "33.6693057958454_-112.12601933960218_270_0.jpg",
        "source": {
            "database": "Unknown"
        },
        "size": {
            "width": "512",
            "height": "512",
            "depth": "3"
        },
        "segmented": "0",
        "object": [
            {
                "name": "walksignal",
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {
                    "xmin": "85",
                    "xmax": "102",
                    "ymin": "207",
                    "ymax": "270"
                }
            },
            {
                "name": "walksignal",
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {
                    "xmin": "397",
                    "xmax": "426",
                    "ymin": "227",
                    "ymax": "272"
                }
            }
        ]
    }

    assert data == groundtruth_data

    with_labels_path_str = os.path.join(
        "sample_files", "with_extra_labels.xml")
    data = get_data_obj_from_xml(with_labels_path_str)
    convert_labels_to_names(data, label_dict, label_categories_dict)

    groundtruth_data = {
        "folder": "sample_files",
        "filename": "33.37889865057694_-111.60161530847279_90_0.jpg",
        "source": {
            "database": "Unknown"
        },
        "size": {
            "width": "512",
            "height": "512",
            "depth": "3"
        },
        "segmented": "0",
        "object": [
            {
                "name": "walksignal",
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {
                    "xmin": "381",
                    "xmax": "396",
                    "ymin": "231",
                    "ymax": "278"
                }
            }
        ]
    }

    assert data == groundtruth_data


def test_convert_label_dict_to_obj():
    with_labels_path_str = os.path.join("sample_files", "with_labels.xml")
    data = get_data_obj_from_xml(with_labels_path_str)
    convert_labels_to_names(data, label_dict, label_categories_dict)
    data_obj = convert_label_dict_to_obj(data, label_dict)

    groundtruth_data = {
        "image_id": "33.6693057958454_-112.12601933960218_270_0.jpg",
        "groundtruth_boxes": [[207.0, 85.0, 102.0, 270.0], [227.0, 397.0, 426.0, 272.0]],
        "groundtruth_scores": [1.0, 1.0],
        "groundtruth_classes": [1, 1]
    }

    assert data_obj == groundtruth_data

    no_obj_path_str = os.path.join("sample_files", "no_objects.xml")
    data = get_data_obj_from_xml(no_obj_path_str)
    convert_labels_to_names(data, label_dict, label_categories_dict)
    data_obj = convert_label_dict_to_obj(data, label_dict)

    groundtruth_data = {
        "image_id": "33.6491385170296_-112.20791456542071_0_0.jpg",
        "groundtruth_boxes": [],
        "groundtruth_scores": [],
        "groundtruth_classes": []
    }

    assert data_obj == groundtruth_data
