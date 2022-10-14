import torch.nn.functional as F
from typing import List, Tuple
import torch.nn.init
from torch import nn, Tensor
from utils.image_list import ImageList


class RPNHead(nn.Module):
    """
    part of PRN head with classification and regression
    """
    def __init__(self, in_channels, num_anchors):
        """
        initialization
        :param in_channels: input channels
        :param num_anchors:  number of anchors
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size = 1, stride = 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size = 1, stride = 1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std = 0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        forward propagation
        :param x: feature map
        :return: result of classification and regression
        """
        logits,  bbox_reg = [], []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class AnchorsGenerator(nn.Module):
    """
    generate anchors
    """
    def __init__(self, sizes = (128, 256, 512), aspect_ratios = (0.5, 1.0, 2.0)):
        """
        initialization
        :param sizes: anchor sizes
        :param aspect_ratios: anchor aspect ratios
        """
        super(AnchorsGenerator, self).__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s, ) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios, ) * len(sizes)

        assert len(sizes) == len(aspect_ratios), "tuple of sizes length is {}, " \
                                                 "but aspect_ratios length is {}".format(len(sizes), len(aspect_ratios))

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype = torch.float32, device = torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        compute anchor sizes
        :param scales: sqrt(anchor_area)
        :param aspect_ratios: h / w ratios
        :param dtype: float32
        :param device: cpu/gpu
        :return: generated base anchors
        """
        scales = torch.as_tensor(scales, dtype = dtype, device = device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype = dtype, device = device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios

        # broadcast
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # left-top right-bottom coordinate relative to anchor center(0, 0)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim = 1) / 2

        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        """
        set anchor coordinate and information
        :param dtype:
        :param device:
        :return: void
        """
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None, "Error: cell anchors is None"
            if cell_anchors[0].deivce == device:
                return

        # generate anchor using sizes and aspect_ratio
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        anchors position in grid coordinate axis map into origin map
        :param grid_sizes:
        :param strides:
        :return:
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        for size, stride, base_achors in zip(grid_sizes, strides, cell_anchors):
            grid_height = grid_width = size
            stride_height, stride_width = stride
            device = base_achors.device

            # for output anchor, compute center
            shifts_x = torch.arange(0, grid_width, dtype = torch.float32, device = device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype = torch.float32, device = device) * stride_height

            # compute each point in predicted feature matrix related to coordinate of origin image
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # for every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim = 1)
            shifts_anchor = shifts.view(-1, 1, 4) + base_achors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List], List[List[List[Tensor]]]) -> List[Tensor]
        """
        cache all anchors information
        :param grid_sizes:
        :param strides:
        :return:
        """
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        """
        forward propagation
        :param image_list: images, and it transformed information
        :param feature_maps: image feature map through backbone
        :return: generated anchor
        """
        # compute one step in feature map equate n pixel stride in origin image
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype = torch.int64, device = device),
                    torch.tensor(image_size[0] // g[1], dtype = torch.int64, device = device)]
                   for g in grid_sizes]
        self.set_cell_anchors(dtype, device)

        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = torch.jit.annotate(List[List[Tensor]], [])
        for i, _ in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        self._cache.clear()
        return anchors