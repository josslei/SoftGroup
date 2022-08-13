import torch
import numpy as np
import math

# @ret (points, color, semantic_label, instance_label)
def loadPth(filename: str) -> tuple:
    data = torch.load(filename)
    return data

def getInstanceInfo(xyz, instance_label, semantic_label):
    pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
    instance_pointnum = []
    instance_cls = []
    instance_num = int(instance_label.max()) + 1
    for i_ in range(instance_num):
        inst_idx_i = np.where(instance_label == i_)
        xyz_i = xyz[inst_idx_i]
        pt_mean[inst_idx_i] = xyz_i.mean(0)
        instance_pointnum.append(inst_idx_i[0].size)
        cls_idx = inst_idx_i[0][0]
        instance_cls.append(semantic_label[cls_idx])
    pt_offset_label = pt_mean - xyz
    return instance_num, instance_pointnum, instance_cls, pt_offset_label

def dataAugment(xyz, jitter=False, flip=False, rot=False, prob=1.0):
    m = np.eye(3)
    if jitter and np.random.rand() < prob:
        m += np.random.randn(3, 3) * 0.1
    if flip and np.random.rand() < prob:
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
    if rot and np.random.rand() < prob:
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                          [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    else:
        # Empirically, slightly rotate the scene can match the results from checkpoint
        theta = 0.35 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                          [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    return np.matmul(xyz, m)

def getXYZMiddle(xyz): #transform_test(xyz, rgb, semantic_label, instance_label, voxel_cfg_scale):
    xyz_middle = dataAugment(xyz, False, False, False)
    #xyz = xyz_middle * voxel_cfg_scale
    #xyz -= xyz.min(0)
    #valid_idxs = np.ones(xyz.shape[0], dtype=bool)
    #instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
    return xyz_middle

