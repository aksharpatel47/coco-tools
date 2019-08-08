from .dataset import ImageDataSet, ImageCollection
from typing import List
from .tf_record import write_tf_record
import requests


def prepare_records(collections: List[ImageCollection], labels_url: str):
    label_path = "labels.pbtxt"
    f = requests.get(labels_url)
    open(label_path, "wb").write(f.content)

    for collection in collections:
        write_tf_record(collection.datasets, label_path,
                        f"{collection.name}.record")


if __name__ == "__main__":
    prefix = "fbike"
    
    collections = [
        ImageCollection(
         name = f"{prefix}_training", 
         datasets = [
           ImageDataSet(
               "Phoennix_BikeSymbol_training",
               "https://www.dropbox.com/s/swazixfufqzhu91/BikesymbolTraining.zip?dl=1",
               ["default", "flipped"]
           ),
         ]
     ),
     ImageCollection(
         name = f"{prefix}_validation", 
         datasets = [
           ImageDataSet(
               "Phoenix_BikeSymbol_validation",
               "https://www.dropbox.com/s/gfdlrvej995bksi/BikeSymbolTest.zip?dl=1",
               ["default"],
               is_ground_truth = True
           )
         ]
     ),
     ImageCollection(
         name = f"{prefix}_test", 
         datasets = [
           ImageDataSet(
               "Phoenix_BikeSymbol_inference",
               "https://www.dropbox.com/s/gfdlrvej995bksi/BikeSymbolTest.zip?dl=1",
               ["default"]
           )
         ]
    ),
    ]

    # collections = [
    #     ImageCollection("infer", [
    #         ImageDataSet(
    #             "sample_files/sample_infer_dataset",
    #             "none",
    #             ["default"]
    #         ),
    #     ])
    # ]

    prepare_records(
        collections, "https://www.dropbox.com/s/nry88tesw5ae6zi/bikelabels.pbtxt?dl=1")
