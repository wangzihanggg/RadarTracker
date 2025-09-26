# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT17 sequence dataset.
"""
import configparser
import csv
import os
from pathlib import Path
import os.path as osp
from argparse import Namespace
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..coco import make_coco_transforms
from ..transforms import Compose


class DemoSequence(Dataset):
    """DemoSequence (MOT17) Dataset.
    """

    def __init__(self, root_dir: str = 'data', img_transform: Namespace = None) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons
                                   above which they are selected
        """
        super().__init__()
        root_dir = '/home/seu2/a_back/wzhdata/724/RadarTracker/data/custom_dataset/train/724_seq1/img1'
        self._data_dir = Path(root_dir)
        assert self._data_dir.is_dir(), f'data_root_dir:{root_dir} does not exist.'

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))

        self.data = self._sequence()
        self.no_gt = True

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self._data_dir.name

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        img, _ = self.transforms(img)
        width, height = img.size(2), img.size(1)
        sample = {}
        sample['img'] = img
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['dets'] = torch.tensor([])
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])

        return sample

    def get_track_boxes_and_visbility(self) -> Tuple[dict, dict]:
        """ Load ground truth boxes and their visibility."""
        boxes = {}
        visibility = {}

        for i in range(1, 611):
            boxes[i] = {}
            visibility[i] = {}

        gt_file = '/home/seu2/a_back/wzhdata/724/RadarTracker/data/custom_dataset/train/724_seq1/gt/gt.txt'
        if not osp.exists(gt_file):
            return boxes, visibility

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # class person, certainity 1
                # if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= 0.5:
                    # Make pixel indexes 0-based, should already be 0-based (or not)
                    x1 = int(row[2]) - 1
                    y1 = int(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + int(row[4]) - 1
                    y2 = y1 + int(row[5]) - 1
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

                    frame_id = int(row[0])
                    track_id = int(row[1])

                    boxes[frame_id][track_id] = bbox
                    visibility[frame_id][track_id] = float(row[8])

        return boxes, visibility

    def _sequence(self) -> List[dict]:
        total = []
        i = 1
        boxes, visibility = self.get_track_boxes_and_visbility()

        for filename in sorted(os.listdir(self._data_dir)):
            extension = os.path.splitext(filename)[1]
            if extension in ['.png', '.jpg']:
                total.append({'im_path': osp.join(self._data_dir, filename),
                              'gt': boxes[i],
                              'vis': visibility[i]})

        return total

    def load_results(self, results_dir: str) -> dict:
        return {}

    def write_results(self, results: dict, output_dir: str) -> None:
        """Write the tracks in the format for MOT16/MOT17 sumbission

        results: dictionary with 1 dictionary for every track with
                 {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_file_path = osp.join(output_dir, self._data_dir.name)

        with open(result_file_path, "w") as r_file:
            writer = csv.writer(r_file, delimiter=',')

            for i, track in results.items():
                for frame, data in track.items():
                    x1 = data['bbox'][0]
                    y1 = data['bbox'][1]
                    x2 = data['bbox'][2]
                    y2 = data['bbox'][3]

                    writer.writerow([
                        frame + 1,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1, -1, -1, -1])
