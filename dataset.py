import os
import shutil
import requests
from zipfile import ZipFile
from typing import List


def download_file(url, downloaded_file_name):
  zf = requests.get(url)
  open(downloaded_file_name, "wb").write(zf.content)


class ImageDataSet():
  def __init__(self, folder_name: str, url: str, image_augmentation: List[str]):
    self.folder_name = folder_name
    self.download_file = f"{self.folder_name}.zip"
    self.url = url
    self.image_augmentation = image_augmentation

  def download(self):
    if os.path.isdir(self.folder_name):
      shutil.rmtree(self.folder_name)
    download_file(self.url, self.download_file)
    with ZipFile(self.download_file) as mz:
      mz.extractall(self.folder_name)