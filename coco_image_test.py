from coco_image import InferImage, Image


def test_image_width_height():
    im = InferImage("sample_files/im.jpg")

    height, width = im.get_image_size()

    assert height == 512
    assert width == 512


def test_coco_image_resize():
    im = Image("sample_files/sample_label_dataset/-_cArIA7sXWern1q92capQ_0.jpg")
    im.make_square()
    obj0 = im.data["object"][0]
    bndbox0 = obj0["bndbox"]

    obj1 = im.data["object"][1]
    bndbox1 = obj1["bndbox"]

    assert bndbox0["xmin"] == "1"
    assert bndbox0["xmax"] == "330"
    assert bndbox0["ymin"] == "378"
    assert bndbox0["ymax"] == "560"

    im.resize_image(512, 512)
    # assert bndbox0["xmin"] == "0"
    # assert bndbox0["xmax"] == "264"
    # assert bndbox0["ymin"] == "302"
    # assert bndbox0["ymax"] == "448"

    im.save_image()
    im.export_to_xml()


if __name__ == "__main__":
    test_coco_image_resize()
