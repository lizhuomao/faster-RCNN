"""
draw boxes on image util
"""

import PIL.ImageDraw as ImageDraw
from PIL import ImageColor
import numpy as np
from PIL import Image


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_text(draw, bbox: list, cls: int, score: float, category_index: dict,
              color: str, font: str = 'arial.ttf', font_size = 24):
    """
    draw object bounding box and class information onto image
    :param draw:
    :param bbox:
    :param cls:
    :param score:
    :param category_index:
    :param color:
    :param font:
    :param font_size:
    :return:
    """


def draw_objs(image: Image,
              bboxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = False):
    """
    draw boxes on objects
    :param image: image data
    :param bboxes: bounding boxes
    :param classes: object class information
    :param scores: object probability information
    :param masks: object mask information
    :param category_index: category dictionary
    :param box_thresh: filter value
    :param mask_thresh:
    :param line_thickness: bounding box width
    :param font: font type
    :param font_size: font size
    :param draw_boxes_on_image:
    :param draw_masks_on_image:
    :return: image with drawing bboxes
    """

    # filter information of low probability
    idx = np.greater(scores, box_thresh)
    bboxes = bboxes[idx]
    classes = classes[idx]
    scores = scores[idx]
    if masks is not None:
        masks = masks[idx]
    if len(bboxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]
    if draw_boxes_on_image:
        # Draw all bboxes onto image
        draw = ImageDraw.Draw(image)
        for bbox, cls, score, color in zip(bboxes, classes, scores, colors):
            left, top, right, bottom = bbox
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width = line_thickness, fill = color)
            draw.text(draw, bbox.tolist(), int(cls), float(score), category_index, color,
                      font, font_size)

    return image

