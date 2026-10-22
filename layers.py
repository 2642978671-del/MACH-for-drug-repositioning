import dgl
import math
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch
from torch.nn import init
from utils import get_activation, to_etype_name
from torch.nn.parameter import Parameter


class GCMCLayer(nn.Module):
    def __init__(self, rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',
                 agg_act=None,
                 ini=True, share_user_item_param=False,
                 basis_units=4, device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        self.user_in_units = user_in_units
        self.msg_units = msg_units
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)
        if ini:
            msg_units = msg_units // 3
        self.msg_units = msg_units
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = {}
        subConv = {}
        self.basis_units = basis_units
        self.att = nn.Parameter(th.randn(len(self.rating_vals), basis_units))
        self.basis = nn.Parameter(th.randn(basis_units, user_in_units, msg_units))
        for i, rating in enumerate(rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=False,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units,
                                                    msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
            else:
                self.W_r = None
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, drug_feat=None, dis_feat=None, Two_Stage=False):
        in_feats = {'drug': drug_feat, 'disease': dis_feat}
        mod_args = {}
        self.W = th.matmul(self.att, self.basis.view(self.basis_units, -1))
        self.W = self.W.view(-1, self.user_in_units, self.msg_units)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            mod_args[rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)
            mod_args[rev_rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)

        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        drug_feat = out_feats['drug']
        dis_feat = out_feats['disease']

        if in_feats['disease'].shape == dis_feat.shape:
            ufeat = dis_feat.view(dis_feat.shape[0], -1)
            ifeat = drug_feat.view(drug_feat.shape[0], -1)

        drug_feat = self.agg_act(drug_feat)
        drug_feat = self.dropout(drug_feat)
        dis_feat = self.agg_act(dis_feat)
        dis_feat = self.dropout(dis_feat)

        drug_feat = self.ifc(drug_feat)
        dis_feat = self.ufc(dis_feat)

        return drug_feat, dis_feat


class GCMCGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None, Two_Stage=False):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError('External weight is provided while at the same time the'
                                       ' module has defined its own weight parameter. Please'
                                       ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            copy_source = fn.copy_u if hasattr(fn, 'copy_u') else fn.copy_src
            graph.update_all(copy_source('h', 'm'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst


class GCN(nn.Module):
    def __init__(self, features, nhid, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(features, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class FGCN(nn.Module):
    def __init__(self, fdim_drug, fdim_disease, nhid1, nhid2, dropout):
        super(FGCN, self).__init__()
        self.FGCN1 = GCN(fdim_drug, nhid1, nhid2, dropout)
        self.FGCN2 = GCN(fdim_disease, nhid1, nhid2, dropout)
        self.dropout = dropout

    def forward(self, drug_graph, drug_sim_feat, dis_graph, disease_sim_feat):
        emb1 = self.FGCN1(drug_sim_feat, drug_graph)
        emb2 = self.FGCN2(disease_sim_feat, dis_graph)
        return emb1, emb2


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = th.mm(input, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_out=16):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)
        self.tanh_f = nn.ReLU()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden1(x2))
        x = th.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        return z.view(z.size()[0], 1) * h1 + (1 - z).view(z.size()[0], 1) * h2


class TransformerFusion(nn.Module):
    """
    兼容所有PyTorch版本的Transformer融合层
    """

    def __init__(self, d_model, nhead=3, dim_feedforward=128, dropout=0.1):
        super(TransformerFusion, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.branch_weight_layer = nn.Linear(d_model * 2, 2)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        x_flat = x.reshape(x.shape[0], -1)
        branch_weights = F.softmax(self.branch_weight_layer(x_flat), dim=1)
        branch_weights = branch_weights.unsqueeze(-1)
        fused = (x * branch_weights).sum(dim=1)

        return fused, branch_weights


# ==========================================================================
# 多尺度膨胀注意力模块 - 稳定版本
# ==========================================================================

class DilateConv1D(nn.Module):
    """
    1D 膨胀卷积注意力 - 简化稳定版本
    每个分支处理完整维度，避免维度分割问题
    """

    def __init__(self, dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.dim = dim
        padding = dilation * (kernel_size - 1) // 2

        # 膨胀卷积
        self.conv_q = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.conv_k = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.conv_v = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)

        self.scale = dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: x [B, seq_len, dim]
        输出: [B, seq_len, dim]
        """
        B, seq_len, C = x.shape

        # 转换为 [B, C, seq_len] 进行卷积
        x_t = x.transpose(1, 2)  # [B, C, seq_len]

        # QKV 卷积
        q = self.conv_q(x_t).transpose(1, 2)  # [B, seq_len, C]
        k = self.conv_k(x_t).transpose(1, 2)
        v = self.conv_v(x_t).transpose(1, 2)

        # 注意力计算
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, seq_len, seq_len]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)  # [B, seq_len, C]

        return out


class MultiDilateAttention1D(nn.Module):
    """
    1D多尺度膨胀注意力模块 - 稳定版本
    每个膨胀分支处理完整维度，然后融合
    """

    def __init__(self, dim, num_heads=3, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilations=[1, 2]):
        super().__init__()
        self.dim = dim
        self.num_dilations = len(dilations)

        # 多尺度膨胀卷积注意力 - 每个分支处理完整维度
        self.dilate_layers = nn.ModuleList([
            DilateConv1D(
                dim=dim,
                kernel_size=kernel_size,
                dilation=d,
                dropout=attn_drop
            ) for d in dilations
        ])

        # 多尺度融合：将多个分支的输出融合回原始维度
        self.fusion = nn.Sequential(
            nn.Linear(dim * self.num_dilations, dim * 2),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(dim * 2, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        输入: x [N, seq_len, dim]
        输出: [N, seq_len, dim]
        """
        outputs = []
        for dilate_layer in self.dilate_layers:
            out = dilate_layer(x)  # [B, seq_len, dim]
            outputs.append(out)

        # 拼接多尺度输出
        multi_scale = torch.cat(outputs, dim=-1)  # [B, seq_len, dim * num_dilations]

        # 融合回原始维度
        fused = self.fusion(multi_scale)  # [B, seq_len, dim]
        fused = self.norm(fused)

        return fused


class ParallelTransformerDilateFusion(nn.Module):
    """
    并联的 Transformer + DilateFormer 融合模块
    """

    def __init__(self, d_model, nhead=3, dim_feedforward=128, dropout=0.1,
                 dilate_kernel_size=3, dilations=[1, 2]):
        super().__init__()

        # ===== 分支1: Transformer全局注意力 =====
        self.transformer_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_norm1 = nn.LayerNorm(d_model)
        self.transformer_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.transformer_norm2 = nn.LayerNorm(d_model)

        # ===== 分支2: 多尺度膨胀注意力 =====
        self.dilate_attn = MultiDilateAttention1D(
            dim=d_model,
            attn_drop=dropout,
            proj_drop=dropout,
            kernel_size=dilate_kernel_size,
            dilations=dilations
        )
        self.dilate_norm = nn.LayerNorm(d_model)

        # ===== 门控融合层 =====
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )

        # ===== 分支权重计算 =====
        self.branch_weight_layer = nn.Linear(d_model * 2, 2)

    def forward(self, x):
        """
        输入: x [N, 2, d_model]
        输出: (融合特征 [N, d_model], 分支权重 [N, 2, 1])
        """
        # ===== 分支1: Transformer =====
        trans_out, _ = self.transformer_attn(x, x, x)
        trans_out = self.transformer_norm1(x + trans_out)
        trans_out = self.transformer_norm2(trans_out + self.transformer_ffn(trans_out))

        # ===== 分支2: 多尺度膨胀注意力 =====
        dilate_out = self.dilate_attn(x)
        dilate_out = self.dilate_norm(x + dilate_out)

        # ===== 门控融合 =====
        trans_global = trans_out.mean(dim=1)
        dilate_global = dilate_out.mean(dim=1)

        combined = torch.cat([trans_global, dilate_global], dim=-1)
        gate = self.gate_fc(combined)

        fused = gate[:, 0:1] * trans_global + gate[:, 1:2] * dilate_global

        # ===== 分支权重 =====
        branch_input = torch.cat([trans_global, dilate_global], dim=-1)
        branch_weights = F.softmax(self.branch_weight_layer(branch_input), dim=-1)
        branch_weights = branch_weights.unsqueeze(-1)

        return fused, branch_weights


class HybridFusionModule(nn.Module):
    """
    混合融合模块：支持多种融合策略
    """

    def __init__(self, d_model, nhead=3, dim_feedforward=128, dropout=0.1,
                 fusion_mode='parallel', dilations=[1, 2]):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.d_model = d_model

        if fusion_mode == 'parallel':
            self.fusion = ParallelTransformerDilateFusion(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                dilations=dilations
            )
        elif fusion_mode == 'sequential':
            self.transformer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.dilate = MultiDilateAttention1D(
                dim=d_model,
                attn_drop=dropout,
                dilations=dilations
            )
            self.branch_weight_layer = nn.Linear(d_model * 2, 2)
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def forward(self, x):
        if self.fusion_mode == 'parallel':
            return self.fusion(x)
        elif self.fusion_mode == 'sequential':
            out = self.transformer(x)
            out = self.dilate(out)
            out = self.norm(out)
            x_flat = out.reshape(out.shape[0], -1)
            branch_weights = F.softmax(self.branch_weight_layer(x_flat), dim=-1)
            branch_weights = branch_weights.unsqueeze(-1)
            fused = (out * branch_weights).sum(dim=1)
            return fused, branch_weights


# ==========================================================================
# AUROC 增强模块
# ==========================================================================

class ContrastiveFusionModule(nn.Module):
    """
    对比学习增强模块
    """

    def __init__(self, d_model, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2)
        )

    def forward(self, feat1, feat2):
        z1 = F.normalize(self.projector(feat1), dim=-1)
        z2 = F.normalize(self.projector(feat2), dim=-1)

        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        batch_size = z1.size(0)

        # 防止 batch_size 不匹配
        if sim_matrix.size(0) != sim_matrix.size(1):
            return torch.tensor(0.0, device=z1.device)

        labels = torch.arange(batch_size, device=z1.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class EnhancedParallelFusion(nn.Module):
    """
    增强版并联融合模块 - 稳定版本
    """

    def __init__(self, d_model, nhead=3, dim_feedforward=128, dropout=0.1,
                 dilate_kernel_size=3, dilations=[1, 2]):
        super().__init__()
        self.d_model = d_model

        # ===== 分支1: Transformer全局注意力 =====
        self.transformer_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_norm1 = nn.LayerNorm(d_model)
        self.transformer_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.transformer_norm2 = nn.LayerNorm(d_model)

        # ===== 分支2: 多尺度膨胀注意力 =====
        self.dilate_attn = MultiDilateAttention1D(
            dim=d_model,
            attn_drop=dropout,
            proj_drop=dropout,
            kernel_size=dilate_kernel_size,
            dilations=dilations
        )
        self.dilate_norm = nn.LayerNorm(d_model)

        # ===== 跨分支交互注意力 =====
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # ===== 自适应门控融合 =====
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )

        # ===== 特征增强层 =====
        self.feature_enhance = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.enhance_norm = nn.LayerNorm(d_model)

        # ===== 分支权重 =====
        self.branch_weight_layer = nn.Linear(d_model * 2, 2)

        # ===== 对比学习模块 =====
        self.contrastive = ContrastiveFusionModule(d_model)

    def forward(self, x, return_contrast_loss=False):
        """
        输入: x [N, 2, d_model]
        输出: (融合特征 [N, d_model], 分支权重 [N, 2, 1], 对比损失)
        """
        # ===== 分支1: Transformer =====
        trans_out, _ = self.transformer_attn(x, x, x)
        trans_out = self.transformer_norm1(x + trans_out)
        trans_out = self.transformer_norm2(trans_out + self.transformer_ffn(trans_out))

        # ===== 分支2: 多尺度膨胀注意力 =====
        dilate_out = self.dilate_attn(x)
        dilate_out = self.dilate_norm(x + dilate_out)

        # ===== 跨分支交互 =====
        cross_out, _ = self.cross_attn(trans_out, dilate_out, dilate_out)
        cross_out = self.cross_norm(trans_out + cross_out)

        # ===== 获取全局表示 =====
        trans_global = trans_out.mean(dim=1)
        dilate_global = dilate_out.mean(dim=1)
        cross_global = cross_out.mean(dim=1)

        # ===== 三路门控融合 =====
        combined = torch.cat([trans_global, dilate_global, cross_global], dim=-1)
        gate = self.gate_fc(combined)

        fused = (gate[:, 0:1] * trans_global +
                 gate[:, 1:2] * dilate_global +
                 gate[:, 2:3] * cross_global)

        # ===== 特征增强 + 残差 =====
        fused = self.enhance_norm(fused + self.feature_enhance(fused))

        # ===== 分支权重 =====
        branch_input = torch.cat([x[:, 0, :], x[:, 1, :]], dim=-1)
        branch_weights = F.softmax(self.branch_weight_layer(branch_input), dim=-1)
        branch_weights = branch_weights.unsqueeze(-1)

        # ===== 对比损失 =====
        if return_contrast_loss:
            topo_feat = x[:, 0, :]
            sim_feat = x[:, 1, :]
            contrast_loss = self.contrastive(topo_feat, sim_feat)
            return fused, branch_weights, contrast_loss

        return fused, branch_weights


class AUROCOptimizedLoss(nn.Module):
    """
    AUROC 优化损失函数
    """
    def __init__(self, gamma=2.0, alpha=0.25, margin=0.4):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.margin = margin

    def focal_loss(self, pred, target):
        """
        Focal Loss: 关注难分类样本
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

    def auc_margin_loss(self, pred, target):
        """
        AUC Margin Loss: 直接优化正负样本排序
        """
        pos_mask = target == 1
        neg_mask = target == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pos_pred = pred[pos_mask]
        neg_pred = pred[neg_mask]

        # 所有正负样本对的 margin loss
        pos_pred = pos_pred.unsqueeze(1)  # [P, 1]
        neg_pred = neg_pred.unsqueeze(0)  # [1, N]

        # 希望 pos_pred > neg_pred + margin
        diff = neg_pred - pos_pred + self.margin  # [P, N]
        loss = torch.clamp(diff, min=0).mean()

        return loss

    def label_smoothing_loss(self, pred, target, smoothing=0.1):
        """
        标签平滑损失
        """
        smooth_target = target * (1 - smoothing) + 0.5 * smoothing
        return F.binary_cross_entropy_with_logits(pred, smooth_target)

    def forward(self, pred, target, mode='combined'):
        """
        mode: 'focal', 'auc_margin', 'combined', 'bce'
        """
        if mode == 'focal':
            return self.focal_loss(pred, target)
        elif mode == 'auc_margin':
            return self.auc_margin_loss(pred, target)
        elif mode == 'combined':
            focal = self.focal_loss(pred, target)
            auc_margin = self.auc_margin_loss(pred, target)
            return focal + 0.5 * auc_margin
        else:
            return F.binary_cross_entropy_with_logits(pred, target)


class MarginRankingDecoder(nn.Module):
    """
    带边界排序的解码器
    """
    def __init__(self, in_units, hidden_units=64, margin=0.5):
        super().__init__()
        self.margin = margin

        self.encoder = nn.Sequential(
            nn.Linear(in_units * 2, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Linear(hidden_units // 2, 1)
        )

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat

            graph.apply_edges(
                lambda edges: {'score': self.encoder(
                    torch.cat([edges.src['h'], edges.dst['h']], dim=1)
                )}
            )
            scores = graph.edata['score'].squeeze()

        return scores

    def margin_loss(self, pos_scores, neg_scores):
        """
        Margin Ranking Loss
        """
        target = torch.ones_like(pos_scores)
        loss = F.margin_ranking_loss(
            pos_scores, neg_scores, target, margin=self.margin
        )
        return loss


class EnhancedMLPDecoder(nn.Module):
    """
    增强型解码器 - 更深的网络 + 残差连接
    """
    def __init__(self, in_units, hidden_units=128, dropout=0.4):
        super().__init__()

        self.fc1 = nn.Linear(in_units * 2, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units // 2)
        self.fc4 = nn.Linear(hidden_units // 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.norm3 = nn.LayerNorm(hidden_units // 2)

        # 残差连接投影
        self.residual_proj = nn.Linear(in_units * 2, hidden_units)

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat

            graph.apply_edges(
                lambda edges: {'h': torch.cat([edges.src['h'], edges.dst['h']], dim=1)}
            )

            h = graph.edata['h']

            # 残差连接
            residual = self.residual_proj(h)

            h = self.fc1(h)
            h = self.norm1(h)
            h = F.gelu(h)
            h = self.dropout(h)

            h = self.fc2(h) + residual  # 残差连接
            h = self.norm2(h)
            h = F.gelu(h)
            h = self.dropout(h)

            h = self.fc3(h)
            h = self.norm3(h)
            h = F.gelu(h)
            h = self.dropout(h)

            pred = self.fc4(h).squeeze()

        return pred


class MLPDecoder(nn.Module):
    """
    原始MLP解码器
    """
    def __init__(self, in_units, dropout_rate=0.2):
        super(MLPDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        self.lin1 = nn.Linear(2 * in_units, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            graph.apply_edges(udf_u_mul_e)
            out = graph.edata['m']

            out = F.relu(self.lin1(out))
            out = self.dropout(out)

            out = F.relu(self.lin2(out))
            out = self.dropout(out)

            out = self.lin3(out)

        return out

class MultiScaleFeatureEnhance(nn.Module):
    """
    多尺度特征增强模块：
    - 多个分支，每个分支是不同缩放比例的 MLP
    - 可解释为：在通道维度上做多尺度非线性建模
    """
    def __init__(self, d_model, reduction_list=(2, 4, 8), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in reduction_list:
            hidden = max(1, d_model // r)
            self.branches.append(
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, d_model),
                    nn.Dropout(dropout)
                )
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [N, d_model]
        """
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        # 多尺度分支取平均 + 残差
        multi_scale = sum(outs) / len(outs)
        out = self.norm(x + multi_scale)
        return out


class CrossModalFusionModule(nn.Module):
    """
    跨模态注意力 + 多尺度特征增强 融合模块
    输入: x [N, 2, d_model]
        x[:,0,:] = 拓扑模态 (GCMC 输出)
        x[:,1,:] = 相似性模态 (FGCN 输出)
    输出:
        fused: [N, d_model]  融合后的表示
        branch_weights: [N, 2, 1] 两个模态的权重
    """
    ATTENTION_MODES = ("modality_tokens", "legacy_single_key", "none")

    def __init__(
        self,
        d_model,
        nhead=3,
        dropout=0.1,
        reduction_list=(2, 4, 8),
        attention_mode="modality_tokens",
        attention_dropout=None,
        use_gate=True,
        use_msf=True,
    ):
        super().__init__()
        if attention_mode not in self.ATTENTION_MODES:
            raise ValueError(
                f"attention_mode must be one of {self.ATTENTION_MODES}, "
                f"got {attention_mode!r}"
            )
        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by attention heads ({nhead})"
            )

        self.d_model = d_model
        self.attention_mode = attention_mode
        self.use_gate = use_gate
        attention_dropout = dropout if attention_dropout is None else attention_dropout
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(
                f"attention_dropout must be in [0, 1), got {attention_dropout}"
            )

        # "none" is a true parameter-free attention ablation.
        if attention_mode != "none":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=attention_dropout,
                batch_first=True,
            )
        self.norm_topo = nn.LayerNorm(d_model)
        self.norm_sim = nn.LayerNorm(d_model)

        # 多尺度特征增强（共享同一模块，保证两个模态有一致的增强机制）
        self.msf = (
            MultiScaleFeatureEnhance(
                d_model,
                reduction_list=reduction_list,
                dropout=dropout,
            )
            if use_msf
            else nn.Identity()
        )

        # 门控融合：为 topo / sim 分配标量权重。
        if use_gate:
            self.gate_fc = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=-1),  # 输出 [N, 2]
            )

    def forward(self, x, return_diagnostics=False):
        """
        x: [N, 2, d_model]
        """
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != self.d_model:
            raise ValueError(
                f"expected x with shape [N, 2, {self.d_model}], got {tuple(x.shape)}"
            )

        attention_weights = None
        if self.attention_mode == "modality_tokens":
            # Each modality token can attend to both topology and similarity.
            attn_out, attention_weights = self.cross_attn(
                x,
                x,
                x,
                need_weights=True,
                average_attn_weights=False,
            )
            topo = self.norm_topo(x[:, 0, :] + attn_out[:, 0, :])
            sim = self.norm_sim(x[:, 1, :] + attn_out[:, 1, :])
        elif self.attention_mode == "legacy_single_key":
            # Historical implementation retained only for reproducibility.
            topo_token = x[:, 0, :].unsqueeze(1)
            sim_token = x[:, 1, :].unsqueeze(1)
            topo_attn, topo_weights = self.cross_attn(
                topo_token,
                sim_token,
                sim_token,
                need_weights=True,
                average_attn_weights=False,
            )
            topo = self.norm_topo(topo_token + topo_attn).squeeze(1)
            sim_attn, sim_weights = self.cross_attn(
                sim_token,
                topo.unsqueeze(1),
                topo.unsqueeze(1),
                need_weights=True,
                average_attn_weights=False,
            )
            sim = self.norm_sim(sim_token + sim_attn).squeeze(1)
            attention_weights = th.cat([topo_weights, sim_weights], dim=2)
        else:
            topo = x[:, 0, :]
            sim = x[:, 1, :]

        # 多尺度特征增强
        topo_enh = self.msf(topo)   # [N, d]
        sim_enh  = self.msf(sim)    # [N, d]

        # 门控融合
        fusion_input = th.cat([topo_enh, sim_enh], dim=-1)  # [N, 2d]
        if self.use_gate:
            gate = self.gate_fc(fusion_input)               # [N, 2]
        else:
            gate = th.full(
                (x.shape[0], 2),
                0.5,
                dtype=x.dtype,
                device=x.device,
            )
        w_topo = gate[:, 0:1]   # [N, 1]
        w_sim  = gate[:, 1:2]

        fused = w_topo * topo_enh + w_sim * sim_enh         # [N, d]
        branch_weights = gate.unsqueeze(-1)                 # [N, 2, 1]

        if return_diagnostics:
            return fused, {
                "attention_weights": attention_weights,
                "gate_weights": gate,
            }
        return fused, branch_weights
# ==========================================================================
# 辅助函数
# ==========================================================================

def udf_u_mul_e_norm(edges):
    return {'reg': edges.src['reg'] * edges.dst['ci']}


def udf_u_mul_e(edges):
    return {'m': th.cat([edges.src['h'], edges.dst['h']], 1)}


def dot_or_identity(A, B, device=None):
    if A is None:
        return B
    elif A.shape[1] == 3:
        if device is None:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1)
        else:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1).to(device)
    else:
        return A
