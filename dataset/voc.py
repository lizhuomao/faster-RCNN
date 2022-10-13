"""
    self-defined voc dataset class
"""

import torch
from PIL import Image
import json
from lxml import etree
import os
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as ts
import random
import transforms
from utils.draw_box import draw_objs


class VOCDataset(Dataset):
    """ voc 2012 dataset """

    def __init__(self, voc_path, year="2012", transforms=None, is_train=True):
        """
        voc dataset init function
        :param voc_path: voc dataset root directory
        :param year: voc dataset with year
        :param transforms: image transform manipulation list
        :param is_train: whether training or evaluating
        """
        # init path
        assert year in ["2012"], "year must be in ['2012']"
        assert os.path.exists(voc_path), "Error: not found {} dataset path".format(voc_path)
        if "VOCdevkit" in voc_path:
            self.path = os.path.join(voc_path, "VOC{}".format(year))
        else:
            self.path = os.path.join(voc_path, "VOCdevkit", "VOC{}".format(year))
        self.img_path = os.path.join(self.path, "JPEGImages")
        self.anno_path = os.path.join(self.path, "Annotations")

        # read xml file
        txt_path = os.path.join(self.path, "ImageSets", "Main",
                                "train.txt" if is_train else "val.txt")
        with open(txt_path) as txt_f:
            xml_path_list = [os.path.join(self.anno_path, "{}.xml".format(line.strip()))
                             for line in txt_f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        for xml_path in xml_path_list:
            if not os.path.exists(xml_path):
                print("Warning: not found {}, skip this annotation file.".format(xml_path))

            data_dict = self.load_xml_file(xml_path)["annotation"]
            if "object" not in data_dict:
                print("INFO: no objects in {}, skip this annotation file.")
            else:
                self.xml_list.append(xml_path)

        assert len(self.xml_list) > 0, "Warning: in this {}.txt file doesn't find any information.".format(
            "train" if is_train else "val")

        # load class dict with json file
        class_json_file = "./dataset/pascal_voc_classes.json"
        assert os.path.exists(class_json_file), "Error: {} file not exist.".format(class_json_file)
        with open(class_json_file) as json_f:
            self.class_dict = json.load(json_f)
        # save image transform manipulations
        self.transforms = transforms

    def __len__(self):
        """
        voc dataset length function
        :return: length of the dataset
        """
        return len(self.xml_list)

    def __getitem__(self, item):
        """
        getting a sample in the voc dataset
        :param item: index of data
        :return: data consist of image and it label
        """
        # load xml dict
        xml_path = self.xml_list[item]
        data_dict = self.load_xml_file(xml_path)["annotation"]
        img_path = os.path.join(self.img_path, data_dict["filename"])

        # load image file
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image {} format not JPEG".format(img_path))

        # load object information
        bboxes, labels, is_difficult, target = \
            [], [], [], {}
        assert "object" in data_dict, "Error: {} lack of object information.".format(xml_path)
        for obj in data_dict["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            if xmax <= xmin or ymax <= ymin:
                print("Warning: in {} xml, there are som bbox w/h <= 0".format(xml_path))
                continue
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                is_difficult.append(int(obj["difficult"]))
            else:
                is_difficult.append(0)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        is_difficult = torch.as_tensor(is_difficult, dtype=torch.int64)
        image_id = torch.tensor([item])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        target["bboxes"] = bboxes
        target["labels"] = labels
        target["is_difficult"] = is_difficult
        target["image_id"] = image_id
        target["area"] = area

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        """
        obtain image height and width
        :param idx: image index
        :return: image height and width
        """
        xml_path = self.xml_list[idx]
        data_dict = self.load_xml_file(xml_path)["annotation"]
        data_height = int(data_dict["size"]["height"])
        data_width = int(data_dict["size"]["width"])
        return data_height, data_width

    def load_xml_file(self, xml_path):
        """
        load xml file
        :param xml_path: xml file path
        :return: xml
        """
        with open(xml_path) as xml_f:
            xml_str = xml_f.read()
        xml = etree.fromstring(xml_str)
        data_dict = self.parse_xml_to_dict(xml)
        return data_dict

    def parse_xml_to_dict(self, xml):
        """
        :param xml: xml tree
        :return: dictionary containing XML contents
        """

        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag == 'object':
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
            else:
                result[child.tag] = child_result[child.tag]
        return {xml.tag: result}


if __name__ == '__main__':
    # load class index
    category_index = {}
    try:
        json_f = open('./dataset/pascal_voc_classes.json', 'r')
        class_dict = json.load(json_f)
        category_index = {str(v): str(k) for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor])
    }

    # load train dataset
    train_dataset = VOCDataset("./", "2012", data_transform["train"], True)
    # draw image with objection detectin information
    _, axes = plt.subplots(2, 3, figsize = (4.5, 4.5))
    axes = axes.flatten()
    for index, ax in zip(random.sample(range(len(train_dataset)), k = 6), axes):
        img, target = train_dataset[index]
        img = ts.ToPILImage()(img)
        plot_img = draw_objs(img, target["bboxes"].numpy(), target["labels"].numpy(),
                             np.ones(target["labels"].shape[0]),
                             category_index = category_index, box_thresh = 0.5,
                             line_thickness = 3, font = 'arial.ttf',
                             font_size = 20)
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()


