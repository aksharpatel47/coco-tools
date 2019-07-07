from .labels import hflip_label, convert_label_dict_to_obj
from object_detection.utils import label_map_util


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


def test_convert_label_dict_to_obj():
    label_dict = label_map_util.get_label_map_dict("sample_labels.pbtxt")
    data = {
        "filename": "anyfile.jpg",
        "object": [
            {"name": "walksignal", "difficult": "0", "bndbox": {
                "xmin": "1", "xmax": "2", "ymin": "0", "ymax": "3"}, }
        ]
    }

    result_data = {
        "image_id": "anyfile.jpg",
        "groundtruth_boxes": [[0.0, 1.0, 2.0, 3.0]],
        "groundtruth_scores": [1.0],
        "groundtruth_classes": [1]
    }

    assert result_data == convert_label_dict_to_obj(data, label_dict)
