import torch
import torch.nn.functional as F
import numpy as np
from pointnet2_ops import pointnet2_utils


def norm_pointcloud(pc):
    # pc: [B, N, 3]
    center = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - center
    m = torch.norm(pc, dim=-1, keepdim=True)
    pc_max = torch.max(m, dim=1, keepdim=True)[0]
    pc = pc / pc_max
    return pc


# ====== rotation error utils ===== #
def relative_rotation_error(gt_rotations, rotations):
    mat = rotations.transpose(-1, -2) @ gt_rotations
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.acos(x)
    return x


# def get_signed_masks(z_lrf, z_pca, x):
#     r"""get the signs for rotation error estimation .
#         Args:
#             z_lrf (Tensor): grouped z axis for LRF (B, N, 3)
#             z_pca (Tensor): repeated z axis for PCA (B, N, 3)
#             x (Tensor): the relative position from center of pca tp lrf (B, N, 3)
#         Returns:
#             mask (Tensor): the mask to decide the signs of each lrf w.r.t. pca (B, N, 3)
#         """
#     B, N, _ = x.size()
#     z_lrf = F.normalize(z_lrf, p=2, dim=-1)
#     z_pca = F.normalize(z_pca, p=2, dim=-1)
#     x = F.normalize(x, p=2, dim=-1)
#     print("normalized xi", x[0, 0])
#
#     inner_product_lrf = (z_lrf * x).sum(dim=-1)
#     inner_product_lrf = inner_product_lrf.clamp(min=-1.0, max=1.0)
#     alpha_lrf = torch.acos(inner_product_lrf)
#     print("alpha angle", alpha_lrf[0, 0] * 180 / np.pi)
#     inner_product_pca = (z_pca * x).sum(dim=-1)
#     inner_product_pca = inner_product_pca.clamp(min=-1.0, max=1.0)
#     alpha_pca = torch.acos(inner_product_pca)
#     print("pca angle", alpha_pca[0, 0] * 180 / np.pi)
#     mask = torch.ones((B, N), device=x.device)
#     # print(torch.unique((alpha_lrf < alpha_pca) == False))
#     mask[alpha_lrf < alpha_pca] = -1
#     return mask.unsqueeze(-1)


def get_signed_masks(z_lrf, z_pca):
    r"""get the signs for rotation error estimation .
        Args:
            z_lrf (Tensor): grouped z axis for LRF (B, N, 3)
            z_pca (Tensor): repeated z axis for PCA (B, N, 3)
        Returns:
            mask (Tensor): the mask to decide the signs of each lrf w.r.t. pca (B, N, 3)
        """
    B, N, _ = z_lrf.size()
    inner_product = (z_lrf * z_pca).sum(dim=-1)
    mask = torch.ones((B, N), device=z_lrf.device)
    mask[inner_product < 0] = -1
    return mask


# ====== basic utils ===== #
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)  # [B, S, 2K]
    return group_idx, sqrdists


def index_points(points, idx) -> torch.Tensor:
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]/[B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # [B, 1]/[B, 1, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # [1, S]/[1, S, K]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


# ====== module utils ===== #
def sample_and_group_lrf(S, k, xyz, points, basis=None, fps_idx=None, knn_idx=None, mode="dynamic", use_xyz=False):
    xyz = xyz.contiguous()
    if fps_idx is None:
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, S).long()  # [B, S]
    new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
    new_points = index_points(points, fps_idx)  # [B, S, C]

    pos_knn_idx = knn_point(k, xyz, new_xyz)[0]  # [B, S, K]
    if knn_idx is None:
        if mode == "static":
            knn_idx = pos_knn_idx
        elif mode == 'dynamic':
            knn_idx = knn_point(k, points, new_points)[0]  # [B, S, K]

    grouped_points = index_points(points, knn_idx)  # [B, S, K, C]
    grouped_xyz = index_points(xyz, pos_knn_idx)  # [B, S, K, 3]
    new_basis = index_points(basis, fps_idx)  # [B, S, 3, 3]

    if use_xyz:
        new_points = torch.cat([grouped_xyz @ new_basis,
                                grouped_points - new_points.unsqueeze(2),
                                new_points.unsqueeze(2).repeat(1, 1, k, 1)], dim=-1)
    else:
        new_points = torch.cat([grouped_points - new_points.unsqueeze(2),
                                new_points.unsqueeze(2).repeat(1, 1, k, 1)], dim=-1)

    return new_xyz, new_points, new_basis, fps_idx, knn_idx


def sample_and_group_pca(S, k, xyz, points, fps_idx=None, knn_idx=None, mode="dynamic", use_xyz=False):
    xyz = xyz.contiguous()
    if fps_idx is None:
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, S).long()  # [B, S]
    new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
    new_points = index_points(points, fps_idx)  # [B, S, C]
    pos_knn_idx = knn_point(k, xyz, new_xyz)[0]  # [B, S, K]
    if knn_idx is None:
        if mode == "static":
            knn_idx = pos_knn_idx
        elif mode == 'dynamic':
            knn_idx = knn_point(k, points, new_points)[0]  # [B, S, K]

    grouped_points = index_points(points, knn_idx)  # [B, S, K, C]
    grouped_xyz = index_points(xyz, pos_knn_idx)  # [B, S, K, 3]

    if use_xyz:
        new_points = torch.cat([grouped_xyz,
                                grouped_points - new_points.unsqueeze(2),
                                new_points.unsqueeze(2).repeat(1, 1, k, 1)], dim=-1)
    else:
        new_points = torch.cat([grouped_points - new_points.unsqueeze(2),
                                new_points.unsqueeze(2).repeat(1, 1, k, 1)], dim=-1)

    return new_xyz, new_points, fps_idx, knn_idx


def get_rotation_invariant_feature(xyz, pca_xyz, S, k, normal=None, use_pca_xyz=False, group_feat=False):
    # S: downsampled point number
    # k: k nearest neighbor number
    # xyz: [B, N, 3], input point coordinates
    # pca_xyz: [B, N, 3], input PCA coordinates
    assert xyz.size()[-1] == 3, "invalid input xyz shape!"
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, S).long()  # [B, S]
    new_xyz = index_points(xyz, fps_idx)  # (B, S, 3)
    if normal is not None:
        normal = index_points(normal, fps_idx)
    lrf_feat, lrf_basis, knn_idx = get_local_reference_frame(xyz, new_xyz, k, use_weighted_norm=True, use_uij=False)

    if use_pca_xyz:
        fps_idx = pointnet2_utils.furthest_point_sample(pca_xyz, S).long()  # (B, S)
        new_pca_xyz = index_points(pca_xyz, fps_idx)
        knn_idx = knn_point(k, xyz=pca_xyz, new_xyz=new_pca_xyz)[0]  # (B, S, K)
        pca_feat = index_points(pca_xyz, knn_idx)
    else:
        new_pca_xyz = index_points(pca_xyz, fps_idx)
        pca_feat = index_points(pca_xyz, knn_idx)

    if group_feat:
        lrf_feat = torch.cat([lrf_feat, lrf_feat], dim=-1)  # (B, S, K, 6)
        pca_feat = torch.cat([new_pca_xyz.unsqueeze(-2).repeat(1, 1, k, 1),
                              pca_feat - new_pca_xyz.unsqueeze(-2)], dim=-1)  # (B, S, K, 6)

    return new_xyz, new_pca_xyz, lrf_feat, lrf_basis, pca_feat, fps_idx


# ====== lrf related ===== #
def weighteNorm(center, neighbor):
    rel_pos = neighbor - center.unsqueeze(-2)  # (B, N, K, 3)
    rel_pos_dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (B, N, K, 1)
    rel_pos_max = torch.max(rel_pos, dim=-1, keepdim=True)[0]  # (B, N, K, 1)
    weights = torch.div((rel_pos_max - rel_pos_dist), (rel_pos_max - rel_pos_dist).sum(-2, keepdim=True))  # (B, N, K, 1)
    cov = (weights * rel_pos).transpose(-1, -2) @ rel_pos
    norm_vec = torch.symeig(cov.cpu(), eigenvectors=True).eigenvectors[..., 0]  # ascending order
    return norm_vec.to(center.device)


def getNearNormal(normal, pi, pj):
    B, N, K, _ = pj.shape
    pipj = pj - pi.unsqueeze(-2)  # (B, N, K, 3)
    normal = normal.unsqueeze(-2).expand_as(pipj)  # (B, N, K, 3)
    inner_product = (normal * pipj).sum(dim=-1)  # (B, N, K)
    index = torch.argmin(torch.abs(inner_product), dim=-1)  # (B, N)
    out = index_points(pipj.view(B * N, K, 3), index.view(-1, 1))
    return out.view(B, N, 3)


def lrf_basis_v1(pi, pj, use_weighted_norm=False, use_uij=False):
    B, N, _ = pi.shape
    if use_weighted_norm:
        normal = weighteNorm(pi, pj)
        inner_product = (normal * pi).sum(-1)
        mask = torch.ones((B, N), device=normal.device)
        mask[inner_product < 0] = -1
        normal = normal * mask.unsqueeze(-1)
        z_axis = F.normalize(normal, dim=-1, eps=1e-16)
    else:
        z_axis = F.normalize(pi, dim=-1, eps=1e-16)
    if use_uij:
        pipj = getNearNormal(z_axis, pi, pj)
    else:
        pipj = pj.mean(dim=-2) - pi
    v = pipj - z_axis * (pipj * z_axis).sum(dim=-1, keepdim=True)  # [B, N, 3]
    x_axis = F.normalize(v, dim=-1, eps=1e-16)  # [B, N, 3]
    y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=-1, eps=1e-16)  # [B, N, 3]
    basis = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    # basis = torch.cat([x_axis.unsqueeze(-1), y_axis.unsqueeze(-1), z_axis.unsqueeze(-1)], dim=-1)  # [B, N, 3, 3]
    return basis


def lrf_basis_v2(xyz, centers):
    # xyz: [B, N, 3], centers: [B, N, 3] for each local group
    z_axis = F.normalize(xyz - centers, dim=-1, eps=1e-12)  # [B, N, 3]
    x_axis = xyz - z_axis * torch.sum(xyz * z_axis, dim=-1, keepdim=True)  # [B, N, 1]
    x_axis = F.normalize(x_axis, dim=-1, eps=1e-12)  # [B, N, 3]
    y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=-1, eps=1e-12)  # [B, N, 3]
    basis = torch.cat([x_axis.unsqueeze(-1), y_axis.unsqueeze(-1), z_axis.unsqueeze(-1)], dim=-1)  # [B, N, 3, 3]
    return basis


def lrf_basis_v3(grouped_xyz):
    B, N, K, _ = grouped_xyz.size()
    mean = grouped_xyz.mean(dim=-2, keepdim=True).view(-1, 1, 3)  # [BN, 1, 3]
    grouped_xyz = grouped_xyz.view(B * N, K, -1)  # [BN, K, 3]
    sqrdists = square_distance(mean, grouped_xyz).view(B, N, K)  # [B, N, K]
    near_idx = torch.topk(sqrdists, 1, dim=-1, largest=False, sorted=True)[-1].view(-1, 1)  # [BN, 1]
    far_idx = torch.topk(sqrdists, 1, dim=-1, largest=True, sorted=True)[-1].view(-1, 1)  # [BN, 1]
    beta_f = index_points(grouped_xyz, far_idx).view(B, N, 3)
    beta_n = index_points(grouped_xyz, near_idx).view(B, N, 3)
    beta_f = F.normalize(beta_f, dim=-1, eps=1e-6)
    beta_z = F.normalize(torch.cross(beta_f, beta_n), dim=-1, eps=1e-6)
    beta_n = F.normalize(torch.cross(beta_n, beta_f), dim=-1, eps=1e-6)
    basis = torch.cat([beta_f.unsqueeze(-1), beta_n.unsqueeze(-1), beta_z.unsqueeze(-1)], dim=-1)  # [B, N, 3, 3]
    return basis


def get_local_reference_frame(xyz, new_xyz, k, use_weighted_norm=False, use_uij=False):
    idx = knn_point(k, xyz=xyz, new_xyz=new_xyz)[0]  # [B, S, K]
    # idx = idx[..., 1:]
    grouped_xyz = index_points(xyz, idx)  # [B, S, K, 3]
    lrf_basis = lrf_basis_v1(new_xyz, grouped_xyz, use_weighted_norm, use_uij)
    data = (grouped_xyz - new_xyz.unsqueeze(-2)) @ lrf_basis
    return data, lrf_basis, idx


# ====== pca related ===== #
def global_pca_adjust(eigen, x, mode='random_v1'):
    # x: [B, S, 3]
    # eigen: [B, 3, 3]
    arr = []
    diff = []
    batch_size = x.size()[0]

    for i in range(3):
        eig = eigen[:, :, i].unsqueeze(-1)  # [B, 3, 1]
        eig_reversed = eig * (-1.0)  # [B, 3, 1]

        A = torch.ge(x @ eig, 0)  # [B, N, 1]
        B = torch.ge(x @ eig_reversed, 0)  # [B, N, 1]

        A_sum = torch.sum(A, dim=1, keepdim=True)  # [B, 1, 1]
        B_sum = torch.sum(B, dim=1, keepdim=True)  # [B, 1, 1]
        A_B_diff = (A_sum - B_sum).squeeze()  # [B]
        diff.append(A_B_diff)
        cond = (B_sum >= A_sum).repeat(1, 3, 1)  # [B, 3, 1]
        arr.append(torch.where(cond, eig_reversed, eig))  # easy to have invalid rotation matrix here

    arr = torch.cat([arr[0], arr[1], arr[2]], dim=-1)  # [B, 3, 3]
    det = torch.det(arr)  # [B] -> check the determinant

    if mode == 'random_v1':
        pass
    elif mode == 'random_v2':
        mask_neg = (det < 1e-6).float()  # invalid rotation
        mask_det_neg = (mask_neg.view(*det.size(), 1, 1).expand(arr.size())) * arr
        mask_det_pos = arr - mask_det_neg
        I = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], device=x.device)
        I = I.expand(*det.size(), 3, 3)
        arr = mask_det_neg @ I + mask_det_pos
    elif mode == 'nearest_v1':
        diff = torch.stack(diff, dim=-1)  # [B, 3] -> [delta_v1, delta_v2, delta_v3]
        idx = torch.min(torch.abs(diff), dim=-1)[1]  # [B]
        idx_base = torch.arange(0, batch_size, device=x.device) * 3
        idx = idx.view(-1) + idx_base
        dir_flip = torch.ones((*idx.size(), 3), device=x.device)  # [B, 3]
        dir_flip.view(-1)[idx] = -1
        dir_flip = dir_flip.view(batch_size, 1, 3)  # [B, 1, 3]

        mask_neg = (det < 1e-6).float()  # invalid rotation matrix
        mask_det_neg = (mask_neg.view(*det.size(), 1, 1).expand(arr.size())) * arr
        mask_det_pos = arr - mask_det_neg
        I = torch.eye(3, device=x.device)
        I = dir_flip * I.expand(*det.size(), 3, 3)
        arr = mask_det_neg @ I + mask_det_pos
    elif mode == 'nearest_v2':
        diff = torch.stack(diff, dim=-1)  # [B, 3] -> [delta_v1, delta_v2, delta_v3]
        _, idx = torch.topk(torch.abs(diff), 2, dim=-1, largest=False)  # return the least 2 values: [B, 2]
        idx = idx.view(-1, 2)  # [B, 2]
        a = torch.arange(3, device=x.device).view(1, 3).repeat(batch_size, 1)  # [B, 3]
        batch_idx = torch.arange(batch_size, device=x.device)  # [B]
        a[batch_idx, idx[:, 1]], a[batch_idx, idx[:, 0]] = a[batch_idx, idx[:, 0]], a[batch_idx, idx[:, 1]]
        I = torch.eye(3, device=x.device).expand(batch_size, 3, 3)
        batch_idx = batch_idx.view(-1, 1).repeat(1, 3)  # [B, 3]

        I = I[batch_idx, :, a].view(batch_size, 3, 3)  # [B, 3, 3]

        mask_neg = (det < 1e-6).float()  # invalid rotation
        mask_det_neg = (mask_neg.view(*det.size(), 1, 1).expand(arr.size())) * arr
        mask_det_pos = arr - mask_det_neg
        arr = mask_det_neg @ I + mask_det_pos

    return arr


def PCA(xyz, adjust=False, mode='random_v1'):
    """
    :param xyz: [B, N, 3]
    :param adjust:
    :param mode:
    :return: pca basis: [B, 3, 3]
    """
    assert xyz.size()[-1] == 3, "invalid input xyz shape"
    sample_cov = (xyz.transpose(2, 1) @ xyz) / (xyz.size()[1] - 1)  # [B, 3, 3]
    basis = torch.symeig(sample_cov.cpu(), eigenvectors=True)[1]  # [v1, v2, v3]
    basis = torch.flip(basis.to(xyz.device), [-1])  # [B, 3, 3]
    if adjust:
        basis = global_pca_adjust(basis, xyz, mode)
    xyz = xyz @ basis
    return xyz, basis
