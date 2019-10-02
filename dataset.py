import os
import requests
from zipfile import ZipFile
from typing import List
import shutil
import urllib.request as req


def download_file(url, downloaded_file_name):
    req.urlretrieve(url, downloaded_file_name)

class ImageDataSet():
    def __init__(self, folder_name: str, url: str, image_augmentation: List[str], is_ground_truth: bool = False):
        self.folder_name = folder_name
        self.download_file = f"{self.folder_name}.zip"
        self.url = url
        self.image_augmentation = image_augmentation
        self.is_ground_truth = is_ground_truth
        self.download()

    def download(self):
        if not os.path.isdir(self.folder_name):
            download_file(self.url, self.download_file)
            with ZipFile(self.download_file) as mz:
                mz.extractall(self.folder_name)

    def cleanup(self):
        if os.path.isdir(self.folder_name):
            shutil.rmtree(self.folder_name)
            os.remove(self.download_file)


class ImageCollection():
    def __init__(self, name: str, datasets: List[ImageDataSet]):
        self.name = name
        self.datasets = datasets
