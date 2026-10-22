import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from layers import (
    GCMCLayer,
    FGCN,
    MLPDecoder,
    ContrastiveFusionModule,
    CrossModalFusionModule,   # 新增：跨模态融合模块
)
from utils import get_activation


def select_branch_embeddings(
        use_topology_branch,
        use_feature_branch,
        topology_embeddings,
        feature_embeddings,
        fusion,
):
    """Return decoder embeddings for full and AdaDR-aligned single branches."""
    if not use_topology_branch and not use_feature_branch:
        raise ValueError("at least one encoder branch must be enabled")
    if use_topology_branch and not use_feature_branch:
        return topology_embeddings
    if use_feature_branch and not use_topology_branch:
        return feature_embeddings
    if fusion is None:
        raise ValueError("fusion is required when both encoder branches are enabled")

    drug_topology, disease_topology = topology_embeddings
    drug_feature, disease_feature = feature_embeddings
    drug_embeddings, _ = fusion(th.stack([drug_topology, drug_feature], dim=1))
    disease_embeddings, _ = fusion(th.stack([disease_topology, disease_feature], dim=1))
    return drug_embeddings, disease_embeddings


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layers = args.layers
        self._act = get_activation(args.model_activation)
        self.use_feature_branch = getattr(args, 'use_feature_branch', True)
        self.use_topology_branch = getattr(args, 'use_topology_branch', True)
        self.use_dual_branch = self.use_feature_branch and self.use_topology_branch
        if not self.use_feature_branch and not self.use_topology_branch:
            raise ValueError('at least one encoder branch must be enabled')

        # ===== 拓扑卷积分支 TGCN (基于 GCMC) =====
        self.TGCN = nn.ModuleList()
        if self.use_topology_branch:
            self.TGCN.append(GCMCLayer(
                args.rating_vals,
                args.src_in_units,
                args.dst_in_units,
                args.gcn_agg_units,
                args.gcn_out_units,
                args.dropout,
                args.gcn_agg_accum,
                agg_act=self._act,
                share_user_item_param=args.share_param,
                device=args.device
            ))

        self.gcn_agg_accum = args.gcn_agg_accum
        self.rating_vals = args.rating_vals
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units
        self.src_in_units = args.src_in_units

        if self.use_topology_branch:
            for i in range(1, args.layers):
                if args.gcn_agg_accum == 'stack':
                    gcn_out_units = args.gcn_out_units * len(args.rating_vals)
                else:
                    gcn_out_units = args.gcn_out_units
                self.TGCN.append(GCMCLayer(
                    args.rating_vals,
                    args.gcn_out_units,
                    args.gcn_out_units,
                    gcn_out_units,
                    args.gcn_out_units,
                    args.dropout,
                    args.gcn_agg_accum,
                    agg_act=self._act,
                    share_user_item_param=args.share_param,
                    ini=False,
                    device=args.device
                ))

        # ===== 相似性卷积分支 FGCN (基于相似度图) =====
        self.FGCN = FGCN(
            args.fdim_drug,
            args.fdim_disease,
            args.nhid1,
            args.nhid2,
            args.dropout
        ) if self.use_feature_branch else None

        # ===== 跨模态注意力 + 多尺度特征增强 融合模块 =====
        # 输入形状： [N, 2, d_model]，两路模态（拓扑 & 相似性）
        self.fusion = CrossModalFusionModule(
            d_model=args.gcn_out_units,
            nhead=getattr(
                args,
                'attention_heads',
                getattr(args, 'transformer_nhead', 3),
            ),
            dropout=args.dropout,
            reduction_list=getattr(args, 'multiscale', (2, 4, 8)),
            attention_mode=getattr(args, 'attention_mode', 'modality_tokens'),
            attention_dropout=getattr(args, 'attention_dropout', 0.1),
            use_gate=getattr(args, 'use_fusion_gate', True),
            use_msf=getattr(args, 'use_msf', True),
        ) if self.use_dual_branch else None

        # ===== 增强型 MLP 解码器：预测边的打分 =====
        self.decoder = EnhancedMLPDecoder(
            in_units=args.gcn_out_units,
            hidden_units=128,
            dropout=args.dropout
        )

        # ===== 对比学习模块：作为高级一致性约束 =====
        self.contrastive = ContrastiveFusionModule(
            d_model=args.gcn_out_units,
            temperature=getattr(args, 'temperature', 0.07),
        ) if self.use_dual_branch and getattr(args, 'use_contrastive', True) else None

        self.rating_vals = args.rating_vals
        self.use_contrastive = self.contrastive is not None
        self.contrastive_weight = getattr(args, 'contrastive_weight', 0.1)

    def forward(self, enc_graph, dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, disease_sim_feat, dis_feat,
                Two_Stage=False):

        # ===== 1) 拓扑卷积分支 (GCMC) =====
        drug_out, dis_out = None, None
        if self.use_topology_branch:
            for i in range(0, self.layers):
                drug_o, dis_o = self.TGCN[i](enc_graph, drug_feat, dis_feat, Two_Stage)

                if i == 0:
                    drug_out = drug_o
                    dis_out = dis_o
                else:
                    # 累加 / 平滑不同层的输出
                    drug_out += drug_o / float(i + 1)
                    dis_out += dis_o / float(i + 1)

                drug_feat = drug_o
                dis_feat = dis_o

        # ===== 2) 相似性卷积分支 (FGCN) =====
        drug_sim_out, dis_sim_out = None, None
        if self.use_feature_branch:
            drug_sim_out, dis_sim_out = self.FGCN(
                drug_graph, drug_sim_feat,
                dis_graph, disease_sim_feat
            )

        # ===== 3) 融合：跨模态注意力 + 多尺度特征增强 =====
        drug_feats, dis_feats = select_branch_embeddings(
            use_topology_branch=self.use_topology_branch,
            use_feature_branch=self.use_feature_branch,
            topology_embeddings=(drug_out, dis_out) if self.use_topology_branch else None,
            feature_embeddings=(drug_sim_out, dis_sim_out) if self.use_feature_branch else None,
            fusion=self.fusion,
        )

        # ===== 4) 解码：边 (drug, disease) -> 预测分数 =====
        pred_ratings = self.decoder(dec_graph, drug_feats, dis_feats)  # [num_edges, 1]

        # ===== 5) 对比学习损失：只保留这种一致性约束 =====
        if self.use_contrastive and self.training:
            drug_contrast_loss = self.contrastive(drug_out, drug_sim_out)
            dis_contrast_loss = self.contrastive(dis_out, dis_sim_out)
            contrast_loss = (drug_contrast_loss + dis_contrast_loss) * self.contrastive_weight
        else:
            contrast_loss = th.tensor(0.0, device=pred_ratings.device)

        return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, contrast_loss


class EnhancedMLPDecoder(nn.Module):
    """
    增强型解码器 - 更深的网络 + 残差连接
    输入：drug_feat, dis_feat
    输出：每条边的预测 logit（未经过 Sigmoid）
    """
    def __init__(self, in_units, hidden_units=128, dropout=0.3):
        super().__init__()

        self.fc1 = nn.Linear(in_units * 2, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units // 2)
        self.fc4 = nn.Linear(hidden_units // 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.norm2 = nn.LayerNorm(hidden_units)
        self.norm3 = nn.LayerNorm(hidden_units // 2)

        # 残差投影：保证维度匹配
        self.residual_proj = nn.Linear(in_units * 2, hidden_units)

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat

            graph.apply_edges(
                lambda edges: {'h': th.cat([edges.src['h'], edges.dst['h']], dim=1)}
            )

            h = graph.edata['h']  # [num_edges, 2 * in_units]

            # ===== 残差连接 =====
            residual = self.residual_proj(h)  # [num_edges, hidden_units]

            h = self.fc1(h)
            h = self.norm1(h)
            h = F.gelu(h)
            h = self.dropout(h)

            h = self.fc2(h) + residual
            h = self.norm2(h)
            h = F.gelu(h)
            h = self.dropout(h)

            h = self.fc3(h)
            h = self.norm3(h)
            h = F.gelu(h)
            h = self.dropout(h)

            pred = self.fc4(h)  # [num_edges, 1]

            return pred
