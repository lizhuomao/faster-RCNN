"""
draw boxes on image util
"""

import PIL.ImageDraw as ImageDraw
from PIL import ImageColor
import numpy as np
from PIL import Image
from PIL.Image import fromarray
import PIL.ImageFont as ImageFont

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


def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)
    img_to_draw = np.copy(np_image)
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))


def draw_text(draw, bbox: list, cls: int, score: float, category_index: dict,
              color: str, font: str = 'arial.ttf', font_size=24):
    """
    draw object detection information onto image
    :param draw: ImageDraw.Draw()
    :param bbox: bounding box
    :param cls: object class information
    :param score: object probability
    :param category_index: category dictionary
    :param color: box line color
    :param font: font type
    :param font_size: font size
    :return: image with draw information
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = bbox
    display_str = "{} : {}%".format(category_index[str(cls)], int(100 * score))
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    display_str_heights = (1 + 2 * 0.05) * max(display_str_heights)
    if top > display_str_heights:
        text_top = top - display_str_heights
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_heights

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top), ds, fill='black', font=font)
        left += text_width


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
        # Draw all bboxes and object detection onto image
        draw = ImageDraw.Draw(image)
        for bbox, cls, score, color in zip(bboxes, classes, scores, colors):
            left, top, right, bottom = bbox
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=color)
            draw_text(draw, bbox.tolist(), int(cls), float(score), category_index, color,
                      font, font_size)
    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, mask_thresh)

    return image
