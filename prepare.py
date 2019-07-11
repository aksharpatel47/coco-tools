from dataset import ImageDataSet, ImageCollection
from typing import List
from tf_record import write_tf_record
import requests


def prepare_records(collections: List[ImageCollection], labels_url: str):
    label_path = "labels.pbtxt"
    f = requests.get(labels_url)
    open(label_path, "wb").write(f.content)

    for collection in collections:
        write_tf_record(collection.datasets, label_path,
                        f"{collection.name}.record")


if __name__ == "__main__":
    collections = [
        ImageCollection("training", [
            ImageDataSet(
                "WalkSignLabel",
                "https://www.dropbox.com/s/hwpdrib721g34l4/WalkSignLabel.zip?dl=1",
                ["default", "flipped"]
            ),
            ImageDataSet(
                "merged_intersection_inference",
                "https://www.dropbox.com/s/8tkjwy6mzdkw4qw/merged_intersection_inference.zip?dl=1",
                ["flipped"]
            )
        ]),
        ImageCollection("validation", [
            ImageDataSet(
                "merged_intersection_inference",
                "https://www.dropbox.com/s/8tkjwy6mzdkw4qw/merged_intersection_inference.zip?dl=1",
                ["default"],
                is_ground_truth=True
            )
        ])
    ]

    prepare_records(
        collections, "https://www.dropbox.com/s/g2zdgl1mvtrl9s0/walksignal.pbtxt?dl=1")
