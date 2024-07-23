import numpy as np
import torch
import open3d as o3d
import os
import copy
from PIL import Image
import json
# import clip
import pointops
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import networkx as nx

from plyfile import PlyData

SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}

# Define the dictionary mapping NYU40 class IDs to their RGB values and label names for GT data
nyu40_colors_to_class = {
    (174, 199, 232): {"id": 1, "name": "wall"},
    (152, 223, 138): {"id": 2, "name": "floor"},
    (31, 119, 180): {"id": 3, "name": "cabinet"},
    (255, 187, 120): {"id": 4, "name": "bed"},
    (188, 189, 34): {"id": 5, "name": "chair"},
    (140, 86, 75): {"id": 6, "name": "sofa"},
    (255, 152, 150): {"id": 7, "name": "table"},
    (214, 39, 40): {"id": 8, "name": "door"},
    (197, 176, 213): {"id": 9, "name": "window"},
    (148, 103, 189): {"id": 10, "name": "bookshelf"},
    (196, 156, 148): {"id": 11, "name": "picture"},
    (23, 190, 207): {"id": 12, "name": "counter"},
    (178, 76, 76): {"id": 13, "name": "blinds"},
    (247, 182, 210): {"id": 14, "name": "desk"},
    (66, 188, 102): {"id": 15, "name": "shelves"},
    (219, 219, 141): {"id": 16, "name": "curtain"},
    (140, 57, 197): {"id": 17, "name": "dresser"},
    (202, 185, 52): {"id": 18, "name": "pillow"},
    (51, 176, 203): {"id": 19, "name": "mirror"},
    (200, 54, 131): {"id": 20, "name": "floor mat"},
    (92, 193, 61): {"id": 21, "name": "clothes"},
    (78, 71, 183): {"id": 22, "name": "ceiling"},
    (172, 114, 82): {"id": 23, "name": "books"},
    (255, 127, 14): {"id": 24, "name": "refrigerator"},
    (91, 163, 138): {"id": 25, "name": "tv"},
    (153, 98, 156): {"id": 26, "name": "paper"},
    (140, 153, 101): {"id": 27, "name": "towel"},
    (158, 218, 229): {"id": 28, "name": "shower curtain"},
    (100, 125, 154): {"id": 29, "name": "box"},
    (178, 127, 135): {"id": 30, "name": "whiteboard"},
    (120, 185, 128): {"id": 31, "name": "person"},
    (146, 111, 194): {"id": 32, "name": "nightstand"},
    (44, 160, 44): {"id": 33, "name": "toilet"},
    (112, 128, 144): {"id": 34, "name": "sink"},
    (96, 207, 209): {"id": 35, "name": "lamp"},
    (227, 119, 194): {"id": 36, "name": "bathtub"},
    (213, 92, 176): {"id": 37, "name": "bag"},
    (94, 106, 211): {"id": 38, "name": "other structure"},
    (82, 84, 163): {"id": 39, "name": "other furniture"},
    (100, 85, 144): {"id": 40, "name": "other prop"},
    (0, 0, 0): {"id": -1, "name": "Unknown"}
}

ScanNet20_colors_to_class = {
    (174, 199, 232): {"id": 1, "name": "wall", "index": 0},
    (152, 223, 138): {"id": 2, "name": "floor", "index": 1},
    (31, 119, 180): {"id": 3, "name": "cabinet", "index": 2},
    (255, 187, 120): {"id": 4, "name": "bed", "index": 3},
    (188, 189, 34): {"id": 5, "name": "chair", "index": 4},
    (140, 86, 75): {"id": 6, "name": "sofa", "index": 5},
    (255, 152, 150): {"id": 7, "name": "table", "index": 6},
    (214, 39, 40): {"id": 8, "name": "door", "index": 7},
    (197, 176, 213): {"id": 9, "name": "window", "index": 8},
    (148, 103, 189): {"id": 10, "name": "bookshelf", "index": 9},
    (196, 156, 148): {"id": 11, "name": "picture", "index": 10},
    (23, 190, 207): {"id": 12, "name": "counter", "index": 11},
    (247, 182, 210): {"id": 14, "name": "desk", "index": 12},
    (219, 219, 141): {"id": 16, "name": "curtain", "index": 13},
    (255, 127, 14): {"id": 24, "name": "refrigerator", "index": 14},
    (158, 218, 229): {"id": 28, "name": "shower curtain", "index": 15},
    (44, 160, 44): {"id": 33, "name": "toilet", "index": 16},
    (112, 128, 144): {"id": 34, "name": "sink", "index": 17},
    (227, 119, 194): {"id": 36, "name": "bathtub", "index": 18},
    (82, 84, 163): {"id": 39, "name": "other furniture", "index": 19},
    (0, 0, 0): {"id": -1, "name": "Unknown", "index": -1}
}

ScanNet200_colors_to_class = {
    (174, 199, 232): {'id': 1, 'name': 'wall', "index": 0}, 
    (188, 189, 34): {'id': 2, 'name': 'chair', "index": 1}, 
    (152, 223, 138): {'id': 3, 'name': 'floor', "index": 2}, 
    (255, 152, 150): {'id': 4, 'name': 'table', "index": 3}, 
    (214, 39, 40): {'id': 5, 'name': 'door', "index": 4}, 
    (91, 135, 229): {'id': 6, 'name': 'couch', "index": 5}, 
    (31, 119, 180): {'id': 7, 'name': 'cabinet', "index": 6}, 
    (229, 91, 104): {'id': 8, 'name': 'shelf', "index": 7}, 
    (247, 182, 210): {'id': 9, 'name': 'desk', "index": 8}, 
    (91, 229, 110): {'id': 10, 'name': 'office chair', "index": 9}, 
    (255, 187, 120): {'id': 11, 'name': 'bed', "index": 10}, 
    (141, 91, 229): {'id': 13, 'name': 'pillow', "index": 11}, 
    (112, 128, 144): {'id': 14, 'name': 'sink', "index": 12}, 
    (196, 156, 148): {'id': 15, 'name': 'picture', "index": 13}, 
    (197, 176, 213): {'id': 16, 'name': 'window', "index": 14}, 
    (44, 160, 44): {'id': 17, 'name': 'toilet', "index": 15}, 
    (148, 103, 189): {'id': 18, 'name': 'bookshelf', "index": 16}, 
    (229, 91, 223): {'id': 19, 'name': 'monitor', "index": 17}, 
    (219, 219, 141): {'id': 21, 'name': 'curtain', "index": 18}, 
    (192, 229, 91): {'id': 22, 'name': 'book', "index": 19}, 
    (88, 218, 137): {'id': 23, 'name': 'armchair', "index": 20}, 
    (58, 98, 137): {'id': 24, 'name': 'coffee table', "index": 21}, 
    (177, 82, 239): {'id': 26, 'name': 'box', "index": 22}, 
    (255, 127, 14): {'id': 27, 'name': 'refrigerator', "index": 23}, 
    (237, 204, 37): {'id': 28, 'name': 'lamp', "index": 24}, 
    (41, 206, 32): {'id': 29, 'name': 'kitchen cabinet', "index": 25}, 
    (62, 143, 148): {'id': 31, 'name': 'towel', "index": 26}, 
    (34, 14, 130): {'id': 32, 'name': 'clothes', "index": 27}, 
    (143, 45, 115): {'id': 33, 'name': 'tv', "index": 28}, 
    (137, 63, 14): {'id': 34, 'name': 'nightstand', "index": 29}, 
    (23, 190, 207): {'id': 35, 'name': 'counter', "index": 30}, 
    (16, 212, 139): {'id': 36, 'name': 'dresser', "index": 31}, 
    (90, 119, 201): {'id': 38, 'name': 'stool', "index": 32}, 
    (125, 30, 141): {'id': 39, 'name': 'cushion', "index": 33}, 
    (150, 53, 56): {'id': 40, 'name': 'plant', "index": 34}, 
    (186, 197, 62): {'id': 41, 'name': 'ceiling', "index": 35}, 
    (227, 119, 194): {'id': 42, 'name': 'bathtub', "index": 36}, 
    (38, 100, 128): {'id': 44, 'name': 'end table', "index": 37}, 
    (120, 31, 243): {'id': 45, 'name': 'dining table', "index": 38}, 
    (154, 59, 103): {'id': 46, 'name': 'keyboard', "index": 39}, 
    (169, 137, 78): {'id': 47, 'name': 'bag', "index": 40}, 
    (143, 245, 111): {'id': 48, 'name': 'backpack', "index": 41}, 
    (37, 230, 205): {'id': 49, 'name': 'toilet paper', "index": 42}, 
    (14, 16, 155): {'id': 50, 'name': 'printer', "index": 43}, 
    (196, 51, 182): {'id': 51, 'name': 'tv stand', "index": 44}, 
    (237, 80, 38): {'id': 52, 'name': 'whiteboard', "index": 45}, 
    (138, 175, 62): {'id': 54, 'name': 'blanket', "index": 46}, 
    (158, 218, 229): {'id': 55, 'name': 'shower curtain', "index": 47}, 
    (38, 96, 167): {'id': 56, 'name': 'trash can', "index": 48}, 
    (190, 77, 246): {'id': 57, 'name': 'closet', "index": 49}, 
    (208, 49, 84): {'id': 58, 'name': 'stairs', "index": 50}, 
    (208, 193, 72): {'id': 59, 'name': 'microwave', "index": 51}, 
    (55, 220, 57): {'id': 62, 'name': 'stove', "index": 52}, 
    (10, 125, 140): {'id': 63, 'name': 'shoe', "index": 53}, 
    (76, 38, 202): {'id': 64, 'name': 'computer tower', "index": 54}, 
    (191, 28, 135): {'id': 65, 'name': 'bottle', "index": 55}, 
    (211, 120, 42): {'id': 66, 'name': 'bin', "index": 56}, 
    (118, 174, 76): {'id': 67, 'name': 'ottoman', "index": 57}, 
    (17, 242, 171): {'id': 68, 'name': 'bench', "index": 58}, 
    (20, 65, 247): {'id': 69, 'name': 'board', "index": 59}, 
    (208, 61, 222): {'id': 70, 'name': 'washing machine', "index": 60}, 
    (162, 62, 60): {'id': 71, 'name': 'mirror', "index": 61}, 
    (210, 235, 62): {'id': 72, 'name': 'copier', "index": 62}, 
    (45, 152, 72): {'id': 73, 'name': 'basket', "index": 63}, 
    (35, 107, 149): {'id': 74, 'name': 'sofa chair', "index": 64}, 
    (160, 89, 237): {'id': 75, 'name': 'file cabinet', "index": 65}, 
    (227, 56, 125): {'id': 76, 'name': 'fan', "index": 66}, 
    (169, 143, 81): {'id': 77, 'name': 'laptop', "index": 67}, 
    (42, 143, 20): {'id': 78, 'name': 'shower', "index": 68}, 
    (25, 160, 151): {'id': 79, 'name': 'paper', "index": 69}, 
    (82, 75, 227): {'id': 80, 'name': 'person', "index": 70}, 
    (253, 59, 222): {'id': 82, 'name': 'paper towel dispenser', "index": 71}, 
    (240, 130, 89): {'id': 84, 'name': 'oven', "index": 72}, 
    (123, 172, 47): {'id': 86, 'name': 'blinds', "index": 73}, 
    (71, 194, 133): {'id': 87, 'name': 'rack', "index": 74}, 
    (24, 94, 205): {'id': 88, 'name': 'plate', "index": 75}, 
    (134, 16, 179): {'id': 89, 'name': 'blackboard', "index": 76}, 
    (159, 32, 52): {'id': 90, 'name': 'piano', "index": 77}, 
    (213, 208, 88): {'id': 93, 'name': 'suitcase', "index": 78}, 
    (64, 158, 70): {'id': 95, 'name': 'rail', "index": 79}, 
    (18, 163, 194): {'id': 96, 'name': 'radiator', "index": 80}, 
    (65, 29, 153): {'id': 97, 'name': 'recycling bin', "index": 81}, 
    (177, 10, 109): {'id': 98, 'name': 'container', "index": 82}, 
    (152, 83, 7): {'id': 99, 'name': 'wardrobe', "index": 83}, 
    (83, 175, 30): {'id': 100, 'name': 'soap dispenser', "index": 84}, 
    (18, 199, 153): {'id': 101, 'name': 'telephone', "index": 85}, 
    (61, 81, 208): {'id': 102, 'name': 'bucket', "index": 86}, 
    (213, 85, 216): {'id': 103, 'name': 'clock', "index": 87}, 
    (170, 53, 42): {'id': 104, 'name': 'stand', "index": 88}, 
    (161, 192, 38): {'id': 105, 'name': 'light', "index": 89}, 
    (23, 241, 91): {'id': 106, 'name': 'laundry basket', "index": 90}, 
    (12, 103, 170): {'id': 107, 'name': 'pipe', "index": 91}, 
    (151, 41, 245): {'id': 110, 'name': 'clothes dryer', "index": 92}, 
    (133, 51, 80): {'id': 112, 'name': 'guitar', "index": 93}, 
    (184, 162, 91): {'id': 115, 'name': 'toilet paper holder', "index": 94}, 
    (50, 138, 38): {'id': 116, 'name': 'seat', "index": 95}, 
    (31, 237, 236): {'id': 118, 'name': 'speaker', "index": 96}, 
    (39, 19, 208): {'id': 120, 'name': 'column', "index": 97}, 
    (223, 27, 180): {'id': 121, 'name': 'bicycle', "index": 98}, 
    (254, 141, 85): {'id': 122, 'name': 'ladder', "index": 99}, 
    (97, 144, 39): {'id': 125, 'name': 'bathroom stall', "index": 100}, 
    (106, 231, 176): {'id': 128, 'name': 'shower wall', "index": 101}, 
    (12, 61, 162): {'id': 130, 'name': 'cup', "index": 102}, 
    (124, 66, 140): {'id': 131, 'name': 'jacket', "index": 103}, 
    (137, 66, 73): {'id': 132, 'name': 'storage bin', "index": 104}, 
    (250, 253, 26): {'id': 134, 'name': 'coffee maker', "index": 105}, 
    (55, 191, 73): {'id': 136, 'name': 'dishwasher', "index": 106}, 
    (60, 126, 146): {'id': 138, 'name': 'paper towel roll', "index": 107}, 
    (153, 108, 234): {'id': 139, 'name': 'machine', "index": 108}, 
    (184, 58, 125): {'id': 140, 'name': 'mat', "index": 109}, 
    (135, 84, 14): {'id': 141, 'name': 'windowsill', "index": 110}, 
    (139, 248, 91): {'id': 145, 'name': 'bar', "index": 111}, 
    (53, 200, 172): {'id': 148, 'name': 'toaster', "index": 112}, 
    (63, 69, 134): {'id': 154, 'name': 'bulletin board', "index": 113}, 
    (190, 75, 186): {'id': 155, 'name': 'ironing board', "index": 114}, 
    (127, 63, 52): {'id': 156, 'name': 'fireplace', "index": 115}, 
    (141, 182, 25): {'id': 157, 'name': 'soap dish', "index": 116}, 
    (56, 144, 89): {'id': 159, 'name': 'kitchen counter', "index": 117}, 
    (64, 160, 250): {'id': 161, 'name': 'doorframe', "index": 118}, 
    (182, 86, 245): {'id': 163, 'name': 'toilet paper dispenser', "index": 119}, 
    (139, 18, 53): {'id': 165, 'name': 'mini fridge', "index": 120}, 
    (134, 120, 54): {'id': 166, 'name': 'fire extinguisher', "index": 121}, 
    (49, 165, 42): {'id': 168, 'name': 'ball', "index": 122}, 
    (51, 128, 133): {'id': 169, 'name': 'hat', "index": 123}, 
    (44, 21, 163): {'id': 170, 'name': 'shower curtain rod', "index": 124}, 
    (232, 93, 193): {'id': 177, 'name': 'water cooler', "index": 125}, 
    (176, 102, 54): {'id': 180, 'name': 'paper cutter', "index": 126}, 
    (116, 217, 17): {'id': 185, 'name': 'tray', "index": 127}, 
    (54, 209, 150): {'id': 188, 'name': 'shower door', "index": 128}, 
    (60, 99, 204): {'id': 191, 'name': 'pillar', "index": 129}, 
    (129, 43, 144): {'id': 193, 'name': 'ledge', "index": 130}, 
    (252, 100, 106): {'id': 195, 'name': 'toaster oven', "index": 131}, 
    (187, 196, 73): {'id': 202, 'name': 'mouse', "index": 132}, 
    (13, 158, 40): {'id': 208, 'name': 'toilet seat cover dispenser', "index": 133}, 
    (52, 122, 152): {'id': 213, 'name': 'furniture', "index": 134}, 
    (128, 76, 202): {'id': 214, 'name': 'cart', "index": 135}, 
    (187, 50, 115): {'id': 221, 'name': 'storage container', "index": 136}, 
    (180, 141, 71): {'id': 229, 'name': 'scale', "index": 137}, 
    (77, 208, 35): {'id': 230, 'name': 'tissue box', "index": 138}, 
    (72, 183, 168): {'id': 232, 'name': 'light switch', "index": 139}, 
    (97, 99, 203): {'id': 233, 'name': 'crate', "index": 140}, 
    (172, 22, 158): {'id': 242, 'name': 'power outlet', "index": 141}, 
    (155, 64, 40): {'id': 250, 'name': 'decoration', "index": 142}, 
    (118, 159, 30): {'id': 261, 'name': 'sign', "index": 143}, 
    (69, 252, 148): {'id': 264, 'name': 'projector', "index": 144}, 
    (45, 103, 173): {'id': 276, 'name': 'closet door', "index": 145}, 
    (111, 38, 149): {'id': 283, 'name': 'vacuum cleaner', "index": 146}, 
    (184, 9, 49): {'id': 286, 'name': 'candle', "index": 147}, 
    (188, 174, 67): {'id': 300, 'name': 'plunger', "index": 148}, 
    (53, 206, 53): {'id': 304, 'name': 'stuffed animal', "index": 149}, 
    (97, 235, 252): {'id': 312, 'name': 'headphones', "index": 150}, 
    (66, 32, 182): {'id': 323, 'name': 'dish rack', "index": 151}, 
    (236, 114, 195): {'id': 325, 'name': 'broom', "index": 152}, 
    (241, 154, 83): {'id': 331, 'name': 'guitar case', "index": 153}, 
    (133, 240, 52): {'id': 342, 'name': 'range hood', "index": 154}, 
    (16, 205, 144): {'id': 356, 'name': 'dustpan', "index": 155}, 
    (75, 101, 198): {'id': 370, 'name': 'hair dryer', "index": 156}, 
    (237, 95, 251): {'id': 392, 'name': 'water bottle', "index": 157}, 
    (191, 52, 49): {'id': 395, 'name': 'handicap bar', "index": 158}, 
    (227, 254, 54): {'id': 399, 'name': 'purse', "index": 159}, 
    (49, 206, 87): {'id': 408, 'name': 'vent', "index": 160}, 
    (48, 113, 150): {'id': 417, 'name': 'shower floor', "index": 161}, 
    (125, 73, 182): {'id': 488, 'name': 'water pitcher', "index": 162}, 
    (229, 32, 114): {'id': 540, 'name': 'mailbox', "index": 163}, 
    (158, 119, 28): {'id': 562, 'name': 'bowl', "index": 164}, 
    (60, 205, 27): {'id': 570, 'name': 'paper bag', "index": 165}, 
    (18, 215, 201): {'id': 572, 'name': 'alarm clock', "index": 166}, 
    (79, 76, 153): {'id': 581, 'name': 'music stand', "index": 167}, 
    (134, 13, 116): {'id': 609, 'name': 'projector screen', "index": 168}, 
    (192, 97, 63): {'id': 748, 'name': 'divider', "index": 169}, 
    (108, 163, 18): {'id': 776, 'name': 'laundry detergent', "index": 170}, 
    (95, 220, 156): {'id': 1156, 'name': 'bathroom counter', "index": 171}, 
    (98, 141, 208): {'id': 1163, 'name': 'object', "index": 172}, 
    (144, 19, 193): {'id': 1164, 'name': 'bathroom vanity', "index": 173}, 
    (166, 36, 57): {'id': 1165, 'name': 'closet wall', "index": 174}, 
    (212, 202, 34): {'id': 1166, 'name': 'laundry hamper', "index": 175}, 
    (23, 206, 34): {'id': 1167, 'name': 'bathroom stall door', "index": 176}, 
    (91, 211, 236): {'id': 1168, 'name': 'ceiling light', "index": 177}, 
    (79, 55, 137): {'id': 1169, 'name': 'trash bin', "index": 178}, 
    (182, 19, 117): {'id': 1170, 'name': 'dumbbell', "index": 179}, 
    (134, 76, 14): {'id': 1171, 'name': 'stair rail', "index": 180}, 
    (87, 185, 28): {'id': 1172, 'name': 'tube', "index": 181}, 
    (82, 224, 187): {'id': 1173, 'name': 'bathroom cabinet', "index": 182}, 
    (92, 110, 214): {'id': 1174, 'name': 'cd case', "index": 183},
    (168, 80, 171): {'id': 1175, 'name': 'closet rod', "index": 184}, 
    (197, 63, 51): {'id': 1176, 'name': 'coffee kettle', "index": 185}, 
    (175, 199, 77): {'id': 1178, 'name': 'structure', "index": 186}, 
    (62, 180, 98): {'id': 1179, 'name': 'shower head', "index": 187}, 
    (8, 91, 150): {'id': 1180, 'name': 'keyboard piano', "index": 188}, 
    (77, 15, 130): {'id': 1181, 'name': 'case of water bottles', "index": 189}, 
    (154, 65, 96): {'id': 1182, 'name': 'coat rack', "index": 190}, 
    (197, 152, 11): {'id': 1183, 'name': 'storage organizer', "index": 191}, 
    (59, 155, 45): {'id': 1184, 'name': 'folded chair', "index": 192}, 
    (12, 147, 145): {'id': 1185, 'name': 'fire alarm', "index": 193}, 
    (54, 35, 219): {'id': 1186, 'name': 'power strip', "index": 194}, 
    (210, 73, 181): {'id': 1187, 'name': 'calendar', "index": 195}, 
    (221, 124, 77): {'id': 1188, 'name': 'poster', "index": 196}, 
    (149, 214, 66): {'id': 1189, 'name': 'potted plant', "index": 197}, 
    (72, 185, 134): {'id': 1190, 'name': 'luggage', "index": 198}, 
    (42, 94, 198): {'id': 1191, 'name': 'mattress', "index": 199},
    (0, 0, 0): {"id": -1, "name": "Unknown", "index": -1}
}

class Voxelize(object):
    def __init__(self,
                 voxel_size=0.05,
                 hash_type="fnv",
                 mode='train',
                 keys=("coord", "normal", "color", "label"),
                 return_discrete_coord=False,
                 return_min_coord=False):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(int)
        min_coord = discrete_coord.min(0) * np.array(self.voxel_size)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)   
        if self.mode == 'train':  # train mode
            # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


def overlap_percentage(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    area_intersection = np.sum(intersection)

    area_mask1 = np.sum(mask1)
    area_mask2 = np.sum(mask2)

    smaller_area = min(area_mask1, area_mask2)

    return area_intersection / smaller_area


def remove_samll_masks(masks, ratio=0.8):
    filtered_masks = []
    skip_masks = set()

    for i, mask1_dict in enumerate(masks):
        if i in skip_masks:
            continue

        should_keep = True
        for j, mask2_dict in enumerate(masks):
            if i == j or j in skip_masks:
                continue
            mask1 = mask1_dict["segmentation"]
            mask2 = mask2_dict["segmentation"]
            overlap = overlap_percentage(mask1, mask2)
            if overlap > ratio:
                if np.sum(mask1) < np.sum(mask2):
                    should_keep = False
                    break
                else:
                    skip_masks.add(j)

        if should_keep:
            filtered_masks.append(mask1)

    return filtered_masks


def draw_graph(graph, title, subplot_position):
    plt.subplot(subplot_position)
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=500)

    # Get all edges in the graph
    edges = graph.edges()

    # Draw edge labels for both count and score
    for edge in edges:
        count_common = graph[edge[0]][edge[1]]['count_common']
        cost = graph[edge[0]][edge[1]]['cost']
        count_cost_label = f"Count: {count_common}, Cost: {cost}"
        nx.draw_networkx_edge_labels(graph, pos, edge_labels={edge: count_cost_label})

    plt.title(title)

def plot_accuracy_metrics(accuracy_metrics, scene_name):
    """
    Plot accuracy metrics.
    
    Parameters:
        accuracy_metrics (dict): Dictionary containing overall accuracy and instance-wise accuracy.
    """
    # Extract overall accuracy and instance-wise accuracy
    overall_accuracy = accuracy_metrics['overall_accuracy']
    instance_accuracy = accuracy_metrics['instance_accuracy']

    # Prepare data for plotting
    group_ids = list(instance_accuracy.keys())
    accuracies = list(instance_accuracy.values())
    
    # Add overall accuracy
    group_ids.append('Overall')
    accuracies.append(overall_accuracy)

    # Convert group IDs to string for labeling
    group_labels = [str(group) for group in group_ids]
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(group_labels)), accuracies, color='skyblue')
    plt.xlabel('Group IDs and Overall')
    plt.ylabel('Accuracy Rate')
    plt.title(f'Accuracy Metrics for Group IDs and Overall - Scene: {scene_name}')
    plt.ylim(-0.1, 1.1)  # Accuracy is between 0 and 1
    plt.xticks(ticks=range(len(group_labels)), labels=group_labels, rotation=45)
    plt.show()

def load_ply(file_path):
    """
    Load a PLY file and return the points and their colors.

    :param file_path: Path to the PLY file.
    :return: List of points and their corresponding colors.
    """
    plydata = PlyData.read(file_path)
    vertex_data = plydata['vertex']
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
    return points, colors

def calculate_segmentation_accuracy(coords_, predicted_labels, ground_truth_labels):
    """
    Calculate the segmentation accuracy metrics.
    
    Parameters:
        coords_ (np.ndarray): Coordinates of the points (not used in accuracy calculation).
        predicted_labels (np.ndarray): Predicted group labels.
        ground_truth_labels (np.ndarray): Ground truth group labels.
    
    Returns:
        dict: Dictionary containing overall accuracy and instance-wise accuracy.
    """
    
    # Flatten the group arrays
    predicted_groups = predicted_labels.flatten()
    ground_truth_groups = ground_truth_labels.flatten()

    # Calculate the overlap matrix
    unique_pred = np.unique(predicted_groups)
    unique_gt = np.unique(ground_truth_groups)
    
    overlap_matrix = np.zeros((len(unique_gt), len(unique_pred)), dtype=int)
    
    for i, gt_label in enumerate(unique_gt):
        for j, pred_label in enumerate(unique_pred):
            overlap_matrix[i, j] = np.sum((ground_truth_groups == gt_label) & (predicted_groups == pred_label))
            #print(f"gt_label: {gt_label}, pred_label: {pred_label}, overlap_matrix[{i}, {j}]: {overlap_matrix[i, j]}")
    
    # Find the best match using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
    
    # Create a mapping from predicted to ground truth labels
    label_mapping = {unique_pred[col]: unique_gt[row] for row, col in zip(row_ind, col_ind)}

    # Debug statements
    #print("Unique predicted labels:", unique_pred)
    #print("Unique ground truth labels:", unique_gt)
    #print("Initial label mapping:", label_mapping)

    # Handle unmapped predicted labels
    remaining_pred_labels = set(unique_pred) - set(label_mapping.keys())
    
    while remaining_pred_labels:
        # For each remaining predicted label, find the best matching ground truth label
        for pred_label in list(remaining_pred_labels):
            overlaps = np.array([np.sum((predicted_groups == pred_label) & (ground_truth_groups == gt_label)) for gt_label in unique_gt])
            best_gt_label_index = np.argmax(overlaps)
            best_gt_label = unique_gt[best_gt_label_index]
            
            # Update the mapping
            label_mapping[pred_label] = best_gt_label
            remaining_pred_labels.remove(pred_label)
    
    # Print the updated label mapping
    #print("Final label mapping:", label_mapping)
    
    # Remap predicted labels
    remapped_predicted_groups = np.array([label_mapping.get(label, -1) for label in predicted_groups])
        
    # Check if all predicted labels are in the mapping
    missing_labels = [label for label in predicted_groups if label not in label_mapping]
    if missing_labels:
        print("Missing labels in label mapping:", missing_labels)
        
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(ground_truth_groups, remapped_predicted_groups)
    
    # Calculate instance-wise accuracy
    instance_accuracy = {}
    for instance in unique_gt:
        instance_mask = (ground_truth_groups == instance)
        instance_accuracy[instance] = accuracy_score(
            ground_truth_groups[instance_mask],
            remapped_predicted_groups[instance_mask]
        )
    
    return {
        "overall_accuracy": overall_accuracy,
        "instance_accuracy": instance_accuracy
    }

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    #print("Coord: ", coord)
    #print(coord.size)
    if color is not None:
        color = to_numpy(color)
    #print("Color: ", color)
    #print(color.size)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")

def read_point_cloud(file_path="pc.ply", logger=None):
    
    if not os.path.exists(file_path):
        if logger:
            logger.error(f"File {file_path} does not exist.")
        return None, None

    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd:
        if logger:
            logger.error(f"Failed to read point cloud from {file_path}.")
        return None, None

    coord = np.asarray(pcd.points)
    color = np.asarray(pcd.colors) if np.asarray(pcd.colors).size != 0 else None

    if logger:
        logger.info(f"Read Point Cloud from: {file_path}")

    return coord, color


def remove_small_group(group_ids, th):
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        if count < th:
            result[group_ids == unique_elements[i]] = -1
    
    return result


def pairwise_indices(length):
    return [[i, i + 1] if i + 1 < length else [i] for i in range(0, length, 2)]


def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array


def get_matching_indices(pcd0, pcd1, search_voxel_size, K=None):
    match_inds = []
    scene_coord = torch.tensor(np.asarray(pcd0.points)).cuda().contiguous().float()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda().float()
    gen_coord = torch.tensor(np.asarray(pcd1.points)).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda().float()

    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()[:,0]
    dis = dis.cpu().numpy()[:,0]

    for i in range(scene_coord.shape[0]):
        if dis[i] < search_voxel_size:
            match_inds.append([i,indices[i]])
    
    return match_inds


def visualize_3d(data_dict, text_feat_path, save_path):
    text_feat = torch.load(text_feat_path)
    group_logits = np.einsum('nc,mc->nm', data_dict["group_feat"], text_feat)
    group_labels = np.argmax(group_logits, axis=-1)
    labels = group_labels[data_dict["group"]]
    labels[data_dict["group"] == -1] = -1
    visualize_pcd(data_dict["coord"], data_dict["color"], labels, save_path)


def visualize_pcd(coord, pcd_color, labels, save_path, islabel=False):
    # alpha = 0.5
    if islabel:
        #label_color = np.array([SCANNET_COLOR_MAP_20[label] for label in labels])
        # overlay = (pcd_color * (1-alpha) + label_color * alpha).astype(np.uint8) / 255
        save_point_cloud(coord, labels, save_path)
    else:
        pcd_color = pcd_color / 255
        save_point_cloud(coord, pcd_color, save_path) 

def visualize_2d(img_color, labels, img_size, save_path):
    import matplotlib.pyplot as plt
    # from skimage.segmentation import mark_boundaries
    # from skimage.color import label2rgb
    label_names = ["wall", "floor", "cabinet", "bed", "chair",
           "sofa", "table", "door", "window", "bookshelf",
           "picture", "counter", "desk", "curtain", "refridgerator",
           "shower curtain", "toilet", "sink", "bathtub", "other"]
    colors = np.array(list(SCANNET_COLOR_MAP_20.values()))[1:]
    segmentation_color = np.zeros((img_size[0], img_size[1], 3))
    for i, color in enumerate(colors):
        segmentation_color[labels == i] = color
    alpha = 1
    overlay = (img_color * (1-alpha) + segmentation_color * alpha).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    patches = [plt.plot([], [], 's', color=np.array(color)/255, label=label)[0] for label, color in zip(label_names, colors)]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, fontsize='small')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def visualize_partition(coord, group_id, save_path):
    group_id = group_id.reshape(-1)
    num_groups = group_id.max() + 1
    group_colors = np.random.rand(num_groups, 3)
    group_colors = np.vstack((group_colors, np.array([0,0,0])))
    color = group_colors[group_id]
    save_point_cloud(coord, color, save_path)


def delete_invalid_group(group, group_feat):
    indices = np.unique(group[group != -1])
    group = num_to_natural(group)
    group_feat = group_feat[indices]
    return group, group_feat

def get_labels_from_colors(colors):
    """
    Map RGB colors to class labels.

    :param colors: List of RGB colors.
    :return: List of class labels.
    """
    labels = []
    for color in colors:
        color_tuple = tuple(color)
        if color_tuple in nyu40_colors_to_class:
            labels.append(nyu40_colors_to_class[color_tuple]["id"])
        else:
            labels.append(-1)  # Unknown label
    return np.array(labels, dtype=np.int16)