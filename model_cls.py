import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils_model import PCA, norm_pointcloud, index_points, sample_and_group_lrf, sample_and_group_pca,\
    relative_rotation_error, get_rotation_invariant_feature


class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm


class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = nn.CrossEntropyLoss()

    def get_metric_loss(self, x, ref):
        '''
        :param x: (bs, C)
        :param ref: (bs, C, S)
        :return: loss
        '''
        bs, C, N = ref.size()
        ref = ref.transpose(0, 1).reshape(C, -1)  # [C, bs*S]
        score = torch.matmul(x, ref) * 64.  # [bs, bs*S]
        score = score.view(bs, bs, N).transpose(1, 2).reshape(bs * N, bs)  # [bs*S, bs]
        gt_label = torch.arange(bs, dtype=torch.long, device=x.device).view(bs, 1).expand(bs, N).reshape(-1)
        return self.criteria(score, gt_label)

    def forward(self, x, refs):
        loss = 0.
        for ref in refs:
            loss += self.get_metric_loss(x, ref)
        return loss


class MLP_2d(nn.Module):
    def __init__(self, chs, use_skip=True, activation='leaky_relu'):
        super().__init__()
        self.use_skip = use_skip
        cs = [nn.Conv2d(chs[i], chs[i + 1], kernel_size=1) for i in range(len(chs) - 1)]
        bs = [nn.BatchNorm2d(chs[i + 1]) for i in range(len(chs) - 1)]
        self.conv2ds = nn.ModuleList(cs)
        self.batchNorms = nn.ModuleList(bs)
        if activation == 'leaky_relu':
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU(True)
        if use_skip:
            self.shortcut_conv = nn.Conv2d(chs[0], chs[-1], kernel_size=1)
            self.shortcut_bn = nn.BatchNorm2d(chs[-1])

    def forward(self, x, permute=True):
        if permute:
            x = x.permute(0, 3, 1, 2).contiguous()  # [B, C_in, N, K]
        if self.use_skip:
            shortcut = self.shortcut_bn(self.shortcut_conv(x))
        for i in range(len(self.conv2ds)):
            x = self.batchNorms[i](self.conv2ds[i](x))
            if self.use_skip:
                if i != len(self.conv2ds) - 1:
                    x = self.relu(x)
            else:
                x = self.relu(x)
        if self.use_skip:
            x = x + shortcut
            x = self.relu(x)
        return x


class MLP_1d(nn.Module):
    def __init__(self, chs, activation='leaky_relu'):
        super().__init__()
        cs = [nn.Conv1d(chs[i], chs[i + 1], kernel_size=1) for i in range(len(chs) - 1)]
        bs = [nn.BatchNorm1d(chs[i + 1]) for i in range(len(chs) - 1)]
        self.conv2ds = nn.ModuleList(cs)
        self.batchNorms = nn.ModuleList(bs)
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.act = nn.ReLU(True)

    def forward(self, x):
        for bn, conv in zip(self.batchNorms, self.conv2ds):
            x = self.act(bn(conv(x)))
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError('Sinusoidal positional encoding with odd d_model: {}'.format(d_model))
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()  # [d_model/2]
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        return embeddings.detach()


class AngleEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma):
        super(AngleEmbedding, self).__init__()
        self.sigma = sigma
        self.factor = 180.0 / (np.pi * self.sigma)
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, angles):
        a_indices = angles * self.factor
        embeddings = self.embedding(a_indices)
        embeddings = self.proj(embeddings)

        return embeddings


class SA_Layer(nn.Module):
    def __init__(self, args, channels, learn_weight=False, sigma=15):
        super(SA_Layer, self).__init__()
        self.args = args
        self.learn_weight = learn_weight
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight

        self.k_conv_pca = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv_pca = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv_pca.weight

        if self.learn_weight:
            self.beta_x = nn.Parameter(torch.ones(1), requires_grad=True)
            self.beta_w = nn.Parameter(torch.ones(1), requires_grad=True)

        self.linear = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.softmax = nn.Softmax(dim=-1)
        self.embedding1 = AngleEmbedding(channels // 4, sigma)
        self.embedding2 = AngleEmbedding(channels // 4, sigma)

    def forward(self, x, pca=None, angle_lrf=None, angle_pca=None):
        x_q = self.q_conv(x).permute(0, 2, 1).contiguous()  # (B, N, C)
        energy_lrf = x_q @ self.k_conv(x)
        energy_pca = x_q @ self.k_conv_pca(pca)

        if angle_lrf is not None:
            energy_self = torch.einsum('bnc,bnmc->bnm', x_q, self.embedding1(angle_lrf))
            energy_lrf = energy_lrf + energy_self

        if angle_pca is not None:
            energy_cross = torch.einsum('bnc, bmc->bnm', x_q, self.embedding2(angle_pca))
            energy_pca = energy_pca + energy_cross

        if self.learn_weight:
            energy = self.softmax(self.beta_w * energy_lrf + (1 - self.beta_w) * energy_pca)
        else:
            energy = self.softmax(energy_lrf + energy_pca)
        energy = energy / (1e-12 + energy.sum(dim=1, keepdim=True))  # (B, N, N)

        point_relation = energy[:, self.args.picked_point_num, :].unsqueeze(1)  # (B, 1, N)

        x_lrf = self.v_conv(x) @ energy
        x_pca = self.v_conv_pca(pca) @ energy
        if self.learn_weight:
            x_r = self.beta_x * x_lrf + (1 - self.beta_x) * x_pca
        else:
            x_r = 0.5 * x_lrf + 0.5 * x_pca
        x = x + F.gelu(self.norm(self.linear(x - x_r)))
        return x, point_relation


class SA_block(nn.Module):
    def __init__(self, args, channels=256, learn_weight=False, sigma=15):
        super(SA_block, self).__init__()
        self.args = args
        self.sa1 = SA_Layer(args, channels, learn_weight, sigma)
        self.sa2 = SA_Layer(args, channels, learn_weight, sigma)
        self.sa3 = SA_Layer(args, channels, learn_weight, sigma)
        self.sa4 = SA_Layer(args, channels, learn_weight, sigma)

    def forward(self, lrf, pca, angle_self=None, angle_cross=None):
        # lrf: [B, C_in, N], pca: [B, C_in, N]
        x1, point_relation1 = self.sa1(lrf, pca, angle_self, angle_cross)
        x2, point_relation2 = self.sa2(x1, pca, angle_self, angle_cross)
        x3, point_relation3 = self.sa3(x2, pca, angle_self, angle_cross)
        x4, point_relation4 = self.sa4(x3, pca, angle_self, angle_cross)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


class Model(nn.Module):
    def __init__(self, args, save_embeddings=False, output_channels=40):
        super(Model, self).__init__()
        self.args = args
        self.save_embeddings = save_embeddings
        self.use_contrast = args.use_contrast
        self.conv1 = MLP_2d([6, 64, 128], use_skip=False)
        self.conv11 = MLP_1d([128, 128])
        self.pca_conv1 = MLP_2d([6, 64, 128], use_skip=False)
        self.pca_conv11 = MLP_1d([128, 128])

        self.conv2 = MLP_2d([128 * 2, 256, 256], use_skip=False)
        self.conv21 = MLP_1d([256, 256])
        self.pca_conv2 = MLP_2d([128 * 2, 256, 256], use_skip=False)
        self.pca_conv21 = MLP_1d([256, 256])

        self.sa_block = SA_block(args, 256, learn_weight=True, sigma=args.sigma)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(256 * 5, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.ReLU(True))

        self.cls = nn.Sequential(
            nn.Linear(args.emb_dims, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, output_channels))

        if self.use_contrast:
            self.pred1 = nn.Sequential(
                nn.Conv1d(256, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, 1),
                Normalize(dim=1)
            )
            self.pred2 = nn.Sequential(
                nn.Conv1d(256, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, 1),
                Normalize(dim=1)
            )
            self.pred3 = nn.Sequential(
                nn.Linear(args.emb_dims, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                Normalize(dim=1)
            )

    def forward(self, x, normal=None):
        assert x.size()[-1] == 3, "invalid input size"
        # shift the center to the global origin
        x = norm_pointcloud(x)
        # 1 get global pca
        pca_xyz, pca_basis = PCA(x, adjust=True, mode=self.args.pca_mode)
        # 2. get local feat
        new_xyz, new_pca_xyz, lrf_feat, lrf_basis, pca_feat, fps_idx = get_rotation_invariant_feature(xyz=x,
                                                                                                      pca_xyz=pca_xyz,
                                                                                                      S=512, k=64,
                                                                                                      normal=None,
                                                                                                      use_pca_xyz=True,
                                                                                                      group_feat=True)

        # update lrf and pca features
        lrf_feat1 = self.conv1(lrf_feat, permute=True)
        lrf_feat1 = self.conv11(lrf_feat1.max(-1)[0]).transpose(2, 1).contiguous()  # [B, N//2, 128]
        pca_feat1 = self.pca_conv1(pca_feat, permute=True)
        pca_feat1 = self.pca_conv11(pca_feat1.max(-1)[0]).transpose(2, 1).contiguous()  # [B, N//2, 128]

        # sample and group
        _, lrf_feat2, local_basis, fps_idx = sample_and_group_lrf(S=128, k=self.args.k,
                                                                  xyz=new_xyz,
                                                                  points=lrf_feat1,
                                                                  basis=lrf_basis,
                                                                  mode="static")[:4]
        _, pca_feat2 = sample_and_group_pca(S=128, k=self.args.k,
                                            xyz=new_pca_xyz,
                                            points=pca_feat1,
                                            mode="static")[:2]

        # update lrf and pca features
        lrf_feat2 = self.conv2(lrf_feat2, permute=True)
        lrf_feat2 = self.conv21(lrf_feat2.max(-1)[0])  # [B, 512, N//8]
        pca_feat2 = self.pca_conv2(pca_feat2, permute=True)
        pca_feat2 = self.pca_conv21(pca_feat2.max(-1)[0])  # [B, 512, N//8]

        # get relative rotation error
        pca_basis = pca_basis.unsqueeze(1).repeat(1, 128, 1, 1)
        angle_cross = relative_rotation_error(pca_basis, local_basis)
        angle_self = relative_rotation_error(local_basis.unsqueeze(1), local_basis.unsqueeze(2))

        if self.use_contrast:
            pred_list = []
            pred_list.append(self.pred2(lrf_feat2))  # [B, 512]
            pred_list.append(self.pred1(pca_feat2))  # [B, 512]

        x = self.sa_block(lrf_feat2, pca_feat2, angle_self, angle_cross)  # [B, 1024, N/4]
        x = torch.cat([x, lrf_feat2], dim=1)  # [B, 1280, N/4]
        x = self.conv_fuse(x).max(dim=-1)[0]  # (B, 1024, N//8)

        if self.use_contrast:
            pred_list.append(self.pred3(x))  # [B, 512]

        x = self.cls(x)

        if self.use_contrast and self.training:
            return x, pred_list
        return x
