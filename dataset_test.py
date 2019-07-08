from .dataset import ImageDataSet
import os
import shutil
import glob

def test_image_dataset_download():

    imd = ImageDataSet("WalkSignLabel", "https://www.dropbox.com/s/hwpdrib721g34l4/WalkSignLabel.zip?dl=1", ["default", "flipped"])
    imd.download()

    assert os.path.isdir(imd.folder_name)

    xml_files = glob.glob(os.path.join(imd.folder_name, "**", "*.xml"), recursive=True)

    assert len(xml_files) > 1

    shutil.rmtree(imd.folder_name)
    os.remove(imd.download_file)