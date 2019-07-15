from coco_image import InferImage

def test_image_width_height():
    im = InferImage("sample_files/im.jpg")

    height, width = im.get_image_size()

    assert height == 512
    assert width == 512

if __name__ == "__main__":
    test_image_width_height()