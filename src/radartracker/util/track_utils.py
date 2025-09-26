#########################################
# Still ugly file with helper functions #
#########################################

import os
from collections import defaultdict
from os import path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from cycler import cycler as cy
from matplotlib import colors
from scipy.interpolate import interp1d

matplotlib.use('Agg')


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1],
                                                                        query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2],
                                                                        query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    import colorsys

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colorbar, colors
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                              boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def plot_sequence(tracks, data_loader, output_dir, write_images, generate_attention_maps):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # infinite color loop
    # cyl = cy('ec', COLORS)
    # loop_cy_iter = cyl()
    # styles = defaultdict(lambda: next(loop_cy_iter))

    # cmap = plt.cm.get_cmap('hsv', )
    mx = 0
    for track_id, track_data in tracks.items():
        mx = max(mx, track_id)
    cmap = rand_cmap(100, type='bright', first_color_black=False, last_color_black=False)
    # cmap = rand_cmap(mx, type='bright', first_color_black=False, last_color_black=False)

    # if generate_attention_maps:
    #     attention_maps_per_track = {
    #         track_id: (np.concatenate([t['attention_map'] for t in track.values()])
    #                    if len(track) > 1
    #                    else list(track.values())[0]['attention_map'])
    #         for track_id, track in tracks.items()}
    #     attention_map_thresholds = {
    #         track_id: np.histogram(maps, bins=2)[1][1]
    #         for track_id, maps in attention_maps_per_track.items()}

        # _, attention_maps_bin_edges = np.histogram(all_attention_maps, bins=2)

    for frame_id, frame_data  in enumerate(tqdm.tqdm(data_loader)):
        img_path = frame_data['img_path'][0]
        img = cv2.imread(img_path)[:, :, (2, 1, 0)]
        height, width, _ = img.shape

        fig = plt.figure()
        fig.set_size_inches(width / 96, height / 96)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)

        if generate_attention_maps:
            attention_map_img = np.zeros((height, width, 4))

        for track_id, track_data in tracks.items():
            if frame_id in track_data.keys():
                bbox = track_data[frame_id]['bbox']

                if 'mask' in track_data[frame_id]:
                    mask = track_data[frame_id]['mask']
                    mask = np.ma.masked_where(mask == 0.0, mask)

                    ax.imshow(mask, alpha=0.5, cmap=colors.ListedColormap([cmap(track_id)]))

                    annotate_color = 'white'
                else:
                    # if track_id == 0:
                    #     color = (0.1, 0.1, 0.1, 1.0)
                    # else:
                    #     color = (0.9, 0.9, 0.9, 1.0)
                    color = cmap(track_id)
                    ax.add_patch(
                        plt.Rectangle(
                            (bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            fill=False,
                            linewidth=2.0,
                            color=color,
                        ))

                    annotate_color = cmap(track_id)

                if write_images == 'debug':
                    ax.annotate(
                        f"{track_id} - {track_data[frame_id]['obj_ind']} ({track_data[frame_id]['score']:.2f})",
                        (bbox[0] + (bbox[2] - bbox[0]) / 2.0, bbox[1] + (bbox[3] - bbox[1]) / 2.0),
                        color=annotate_color, weight='bold', fontsize=12, ha='center', va='center')

                if 'attention_map' in track_data[frame_id]:
                    attention_map = track_data[frame_id]['attention_map']
                    attention_map = cv2.resize(attention_map, (width, height))

                    # attention_map_img = np.ones((height, width, 4)) * cmap(track_id)
                    # # max value will be at 0.75 transparency
                    # attention_map_img[:, :, 3] = attention_map * 0.75 / attention_map.max()

                    # _, bin_edges = np.histogram(attention_map, bins=2)
                    # attention_map_img[:, :][attention_map < bin_edges[1]] = 0.0

                    # attention_map_img += attention_map_img

                    # _, bin_edges = np.histogram(attention_map, bins=2)

                    norm_attention_map = attention_map / attention_map.max()

                    high_att_mask = norm_attention_map > 0.25 # bin_edges[1]
                    attention_map_img[:, :][high_att_mask] = cmap(track_id)
                    attention_map_img[:, :, 3][high_att_mask] = norm_attention_map[high_att_mask] * 0.5

                    # attention_map_img[:, :] += (np.tile(attention_map[..., np.newaxis], (1,1,4)) / attention_map.max()) * cmap(track_id)
                    # attention_map_img[:, :, 3] = 0.75

        if generate_attention_maps:
            ax.imshow(attention_map_img, vmin=0.0, vmax=1.0)

        plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        # print(output_dir, osp.basename(img_path))
        plt.savefig(osp.join(output_dir, osp.basename(img_path)), dpi=96)
        plt.close()


def interpolate_tracks(tracks):
    for i, track in tracks.items():
        frames = []
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        for f, data in track.items():
            frames.append(f)
            x0.append(data['bbox'][0])
            y0.append(data['bbox'][1])
            x1.append(data['bbox'][2])
            y1.append(data['bbox'][3])

        if frames:
            x0_inter = interp1d(frames, x0)
            y0_inter = interp1d(frames, y0)
            x1_inter = interp1d(frames, x1)
            y1_inter = interp1d(frames, y1)

            for f in range(min(frames), max(frames) + 1):
                bbox = np.array([
                    x0_inter(f),
                    y0_inter(f),
                    x1_inter(f),
                    y1_inter(f)])
                tracks[i][f]['bbox'] = bbox
        else:
            tracks[i][frames[0]]['bbox'] = np.array([
                x0[0], y0[0], x1[0], y1[0]])

    return interpolated


def bbox_transform_inv(boxes, deltas):
    # Input should be both tensor or both Variable and on the same device
    if len(boxes) == 0:
        return deltas.detach() * 0

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.cat(
        [_.unsqueeze(2) for _ in [pred_ctr_x - 0.5 * pred_w,
                                pred_ctr_y - 0.5 * pred_h,
                                pred_ctr_x + 0.5 * pred_w,
                                pred_ctr_y + 0.5 * pred_h]], 2).view(len(boxes), -1)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    boxes must be tensor or Variable, im_shape can be anything but Variable
    """
    if not hasattr(boxes, 'data'):
        boxes_ = boxes.numpy()

    boxes = boxes.view(boxes.size(0), -1, 4)
    boxes = torch.stack([
        boxes[:, :, 0].clamp(0, im_shape[1] - 1),
        boxes[:, :, 1].clamp(0, im_shape[0] - 1),
        boxes[:, :, 2].clamp(0, im_shape[1] - 1),
        boxes[:, :, 3].clamp(0, im_shape[0] - 1)
    ], 2).view(boxes.size(0), -1)

    return boxes


def get_center(pos):
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


def get_width(pos):
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    return torch.Tensor([[
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2
    ]]).cuda()


def warp_pos(pos, warp_matrix):
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()

def calculate_iou(box1, box2):
    """
    计算两个bbox的IoU
    box: [x, y, width, height]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算交集区域的坐标
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # 计算交集区域面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算并集区域面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def match_bboxes(gt_boxes, track_boxes, gt_ids, track_ids, iou_threshold=0.5):
    """
    将gt_boxes和track_boxes中的bbox一一配对，并返回匹配的ID和未匹配的ID
    :param gt_boxes: numpy.ndarray, shape (N, 4), 格式为 [x, y, width, height]
    :param track_boxes: numpy.ndarray, shape (M, 4), 格式为 [x, y, width, height]
    :param gt_ids: list, shape (N,), gt_boxes的ID
    :param track_ids: list, shape (M,), track_boxes的ID
    :param iou_threshold: IoU阈值，默认0.5
    :return: matched_gt_boxes, matched_track_boxes, matched_gt_ids, matched_track_ids, unmatched_gt_boxes, unmatched_track_boxes, unmatched_gt_ids, unmatched_track_ids
    """
    # 初始化匹配和未匹配的列表
    matched_gt_indices = []
    matched_track_indices = []
    matched_pairs = []

    # 遍历gt_boxes中的每个bbox
    for i, gt_box in enumerate(gt_boxes):
        max_iou = 0
        best_match_index = -1

        # 遍历track_boxes中的每个bbox，寻找最佳匹配
        for j, track_box in enumerate(track_boxes):
            iou = calculate_iou(gt_box, track_box)
            if iou > max_iou:
                max_iou = iou
                best_match_index = j

        # 如果找到匹配的bbox，则配对
        if best_match_index != -1 and max_iou >= iou_threshold:
            matched_pairs.append((i, best_match_index))  # 保存匹配的索引
            matched_gt_indices.append(i)
            matched_track_indices.append(best_match_index)

    # 根据匹配的索引提取匹配的bbox和ID
    matched_gt_boxes = gt_boxes[matched_gt_indices]
    matched_track_boxes = track_boxes[matched_track_indices]
    matched_gt_ids = [gt_ids[i] for i in matched_gt_indices]
    matched_track_ids = [track_ids[j] for j in matched_track_indices]

    # 找到未匹配的gt_boxes和track_boxes及其ID
    unmatched_gt_indices = set(range(len(gt_boxes))) - set(matched_gt_indices)
    unmatched_track_indices = set(range(len(track_boxes))) - set(matched_track_indices)

    unmatched_gt_boxes = gt_boxes[list(unmatched_gt_indices)]
    unmatched_track_boxes = track_boxes[list(unmatched_track_indices)]
    unmatched_gt_ids = [gt_ids[i] for i in unmatched_gt_indices]
    unmatched_track_ids = [track_ids[j] for j in unmatched_track_indices]

    return (
        matched_gt_boxes, matched_track_boxes,
        matched_gt_ids, matched_track_ids,
        unmatched_gt_boxes, unmatched_track_boxes,
        unmatched_gt_ids, unmatched_track_ids
    )


def get_mot_accum(results, seq_loader):
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for frame_id, frame_data in enumerate(seq_loader):
        gt = frame_data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, gt_box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(gt_box[0])

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack(
                (gt_boxes[:, 0],
                 gt_boxes[:, 1],
                 gt_boxes[:, 2] - gt_boxes[:, 0],
                 gt_boxes[:, 3] - gt_boxes[:, 1]), axis=1)
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, track_data in results.items():
            if frame_id in track_data:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(track_data[frame_id]['bbox'])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack(
                (track_boxes[:, 0],
                 track_boxes[:, 1],
                 track_boxes[:, 2] - track_boxes[:, 0],
                 track_boxes[:, 3] - track_boxes[:, 1]), axis=1)
        else:
            track_boxes = np.array([])

        # track_boxes_sorted = track_boxes[track_boxes[:, 0].argsort()]
        # gt_boxes_sorted = gt_boxes[gt_boxes[:, 0].argsort()]
        (
            matched_gt_boxes, matched_track_boxes,
            matched_gt_ids, matched_track_ids,
            unmatched_gt_boxes, unmatched_track_boxes,
            unmatched_gt_ids, unmatched_track_ids
        ) = match_bboxes(gt_boxes, track_boxes, gt_ids, track_ids)

        # 输出结果
        # print("匹配的gt_boxes：")
        # print(matched_gt_boxes)
        #
        # print("\n匹配的track_boxes：")
        # print(matched_track_boxes)
        #
        # print("\n匹配的gt_ids：")
        # print(matched_gt_ids)
        #
        # print("\n匹配的track_ids：")
        # print(matched_track_ids)
        #
        # print("\n未匹配的gt_boxes：")
        # print(unmatched_gt_boxes)
        #
        # print("\n未匹配的track_boxes：")
        # print(unmatched_track_boxes)
        #
        # print("\n未匹配的gt_ids：")
        # print(unmatched_gt_ids)
        #
        # print("\n未匹配的track_ids：")
        # print(unmatched_track_ids)

        distance = mm.distances.iou_matrix(matched_gt_boxes, matched_track_boxes, max_iou=0.5)
        # distance = mm.distances.iou_matrix(gt_boxes_sorted, track_boxes_sorted, max_iou=0.5)
        # print(gt_ids[:len(matched_gt_boxes)])

        # mot_accum.update(
        #     gt_ids,
        #     track_ids,
        #     distance)
        mot_accum.update(
            matched_gt_ids,
            matched_track_ids,
            distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=True):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,)
    return summary, str_summary
