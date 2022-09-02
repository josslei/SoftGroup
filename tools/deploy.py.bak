import argparse

import numpy as np
import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset

from softgroup.model import SoftGroup
from softgroup.util import load_checkpoint
from softgroup.util import (init_dist, get_root_logger)

from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import tools_deploy as tools

POINT_CLOUD_PATH = "../sp-data/preprocessed-raw-point-cloud/0_preprocessed-raw.pth"

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args

def getData():
    coord, feat, semantic_label, instance_label = tools.loadPth(POINT_CLOUD_PATH)
    coord_middle = tools.getXYZMiddle(coord)
    inst_num, inst_pointnum, inst_cls, pt_offset_label = tools.getInstanceInfo(coord_middle, instance_label.astype(np.int32), semantic_label)

    voxel_coords = p2v_map = v2p_map = spatial_shape = torch.from_numpy(np.ascontiguousarray(coord.copy())).float()

    coord = torch.from_numpy(coord).long()
    feat = torch.from_numpy(feat).float()
    semantic_label = torch.from_numpy(semantic_label)
    instance_label = torch.from_numpy(instance_label)
    coord_float = torch.from_numpy(coord_middle)
    # ---
    # softgroup/data/custom.py
    #instance_label[np.where(instance_label != -100)] += total_inst_num
    #total_inst_num += inst_num
    #scan_ids.append(scan_id)
    #coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
    #coords_float.append(coord_float)
    #feats.append(feat)
    #semantic_labels.append(semantic_label)
    #instance_labels.append(instance_label)
    #instance_pointnum.extend(inst_pointnum)
    #instance_cls.extend(inst_cls)
    #pt_offset_labels.append(pt_offset_label)
    #batch_id += 1
    #voxel_coord, v2p_map, p2v_map = tools.voxelization_idx(coord, 0)
    #spatial_shape = np.clip(
    #   coord.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
    return {
        'scan_ids': 1,
        'coords': coord,
        'batch_idxs': 1,
        'voxel_coords': voxel_coords,
        'p2v_map': p2v_map,
        'v2p_map': v2p_map,
        'coords_float': coord_float,
        'feats': feat,
        'semantic_labels': semantic_label,
        'instance_labels': instance_label,
        'instance_pointnum': inst_pointnum,
        'instance_cls': inst_cls,
        'pt_offset_labels': pt_offset_label,
        'spatial_shape': None,
        'batch_size': spatial_shape
    }

if __name__ == "__main__":
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()  # logger creates logs

    model = SoftGroup(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f'Load state dict from {args.checkpoint}')
    # [0] path_to_checkpoint; [1] to write logs; [2] model instance
    load_checkpoint(args.checkpoint, logger, model)

    #dataset = build_dataset(cfg.data.test, logger)
    #dataloader = build_dataloader(dataset, training=False, dist=args.dist, **cfg.dataloader.test)

    # (coord, feat, semantic_label, instance_label)
    # labels are default to be zero

    data = getData()

    with torch.no_grad():
        model.eval()
        result = model(data)
        print(type(result))
        print(result)
