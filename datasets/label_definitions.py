# datasets/label_definitions.py
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class _GTA5LabelDef:
    """Helper dataclass to store GTA5 label properties."""

    name: str
    ID: int
    color: Tuple[int, int, int]


class GTA5LabelInfo:
    """
    Manages GTA5 label definitions and provides a color-to-ID map.
    The IDs are consistent with Cityscapes `trainId` for common classes.
    """

    road = _GTA5LabelDef(name="road", ID=0, color=(128, 64, 128))
    sidewalk = _GTA5LabelDef(name="sidewalk", ID=1, color=(244, 35, 232))
    building = _GTA5LabelDef(name="building", ID=2, color=(70, 70, 70))
    wall = _GTA5LabelDef(name="wall", ID=3, color=(102, 102, 156))
    fence = _GTA5LabelDef(name="fence", ID=4, color=(190, 153, 153))
    pole = _GTA5LabelDef(name="pole", ID=5, color=(153, 153, 153))
    light = _GTA5LabelDef(name="traffic light", ID=6, color=(250, 170, 30))
    sign = _GTA5LabelDef(name="traffic sign", ID=7, color=(220, 220, 0))
    vegetation = _GTA5LabelDef(name="vegetation", ID=8, color=(107, 142, 35))
    terrain = _GTA5LabelDef(name="terrain", ID=9, color=(152, 251, 152))
    sky = _GTA5LabelDef(name="sky", ID=10, color=(70, 130, 180))
    person = _GTA5LabelDef(name="person", ID=11, color=(220, 20, 60))
    rider = _GTA5LabelDef(name="rider", ID=12, color=(255, 0, 0))
    car = _GTA5LabelDef(name="car", ID=13, color=(0, 0, 142))
    truck = _GTA5LabelDef(name="truck", ID=14, color=(0, 0, 70))
    bus = _GTA5LabelDef(name="bus", ID=15, color=(0, 60, 100))
    train = _GTA5LabelDef(name="train", ID=16, color=(0, 80, 100))
    motorcycle = _GTA5LabelDef(name="motorcycle", ID=17, color=(0, 0, 230))
    bicycle = _GTA5LabelDef(name="bicycle", ID=18, color=(119, 11, 32))

    definitions: List[_GTA5LabelDef] = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motorcycle,
        bicycle,
    ]

    # Create a mapping from color tuples to a class ID for efficient conversion
    color_to_id_map = {label_def.color: label_def.ID for label_def in definitions}

    # The ignore index to use for pixels that don't map to any defined color
    ignore_id = 255
