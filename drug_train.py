import os
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

from model import Net
from evaluate import evaluate
from data import DrugDataLoader
from training_protocol import (
    EarlyStoppingTracker,
    fixed_epoch_test_schedule,
    load_completed_fixed_fold,
    parse_positive_int_tuple,
    stratified_inner_split,
)
from utils import MetricLogger, setup_seed  # common_loss 已不再使用


def train(args, dataset, graph_data, cv):
    # 特征维度信息
    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_drug = dataset.drug_feature_shape[0]
    args.fdim_disease = dataset.disease_feature_shape[0]

    # 相似性图 & 特征
    drug_graph = dataset.drug_graph.to(args.device)
    dis_graph = dataset.disease_graph.to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)
    args.rating_vals = dataset.possible_rel_values

    # ===== 构建模型 =====
    model = Net(args=args).to(args.device)

    # ===== 仅使用 BCEWithLogitsLoss 作为监督损失 =====
    rel_loss = nn.BCEWithLogitsLoss()

    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr)
    print("Loading network finished ...\n")

    # ===== 日志 =====
    test_loss_logger = MetricLogger(
        ['iter', 'loss', 'auroc', 'aupr'],
        ['%d', '%.4f', '%.4f', '%.4f'],
        os.path.join(args.save_dir, 'test_metric%d.csv' % args.save_id)
    )

    # ===== 训练数据 =====
    train_gt_ratings = graph_data['train'][2].to(args.device)  # [num_edges]
    train_enc_graph = graph_data['train'][0].int().to(args.device)
    train_dec_graph = graph_data['train'][1].int().to(args.device)
    drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature
    drug_feat = drug_feat.to(args.device)
    dis_feat = dis_feat.to(args.device)

    print("Start training ...")
    start = time.perf_counter()

    # ========== 修改：同时记录 best_auroc 和对应的 best_aupr ==========
    best_auroc = -float('inf')  # 初始化最佳 AUROC
    best_aupr = -float('inf')  # 初始化最佳 AUROC 对应的 AUPR
    best_iter = 0  # 记录最佳结果对应的迭代次数

    for iter_idx in range(1, args.train_max_iter):
        model.train()
        Two_Stage = False

        # ===== 前向传播 =====
        model_output = model(
            train_enc_graph, train_dec_graph,
            drug_graph, drug_sim_feat, drug_feat,
            dis_graph, dis_sim_feat, dis_feat,
            Two_Stage
        )

        if len(model_output) == 6:
            pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out, contrast_loss = model_output
        else:
            pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out = model_output
            contrast_loss = th.tensor(0.0, device=args.device)

        # [num_edges, 1] -> [num_edges]
        pred_ratings = pred_ratings.squeeze(-1)

        # ===== 监督损失（BCE） =====
        base_loss = rel_loss(pred_ratings, train_gt_ratings)

        # ===== 只保留对比学习作为一致性约束 =====
        if isinstance(contrast_loss, th.Tensor):
            loss = base_loss + contrast_loss
        else:
            loss = base_loss

        # ===== 反向传播 =====
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        # ===== 测试评估（按原逻辑，每若干 iter 评一次） =====
        auroc, aupr, y_true, y_score = evaluate(
            args, model, graph_data,
            drug_graph, drug_feat, drug_sim_feat,
            dis_graph, dis_feat, dis_sim_feat
        )

        # Log metrics
        test_loss_logger.log(iter=iter_idx, loss=loss.item(), auroc=auroc, aupr=aupr)

        # ========== 修改：更新 best_auroc 时同时记录对应的 aupr ==========
        if auroc > best_auroc:
            best_auroc = auroc
            best_aupr = aupr  # 记录最佳 AUROC 对应的 AUPR
            best_iter = iter_idx  # 记录最佳迭代次数

        # ========== 修改：日志输出同时显示 best_auroc 和 best_aupr ==========
        logging_str = "Iter={}, loss={:.4f}, AUROC={:.4f}, AUPR={:.4f} | Best: AUROC={:.4f}, AUPR={:.4f} (iter={})".format(
            iter_idx, loss.item(), auroc, aupr, best_auroc, best_aupr, best_iter
        )

        if iter_idx % args.train_valid_interval == 0:
            print("test-logging_str", logging_str)

    end = time.perf_counter()
    print("running time", time.strftime("%H:%M:%S", time.gmtime(round(end - start))))

    # ========== 修改：输出最佳结果包含 AUROC 和 AUPR ==========
    print(f"CV{cv + 1} Training Finished!")
    print(f"Best Result (CV{cv + 1}): AUROC={best_auroc:.4f}, AUPR={best_aupr:.4f} (at iter={best_iter})")

    test_loss_logger.close()

    # ========== 修改：返回该折的最佳 AUROC 和 AUPR ==========
    return best_auroc, best_aupr


def _prepare_model(args, dataset, seed):
    setup_seed(seed)
    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_drug = dataset.drug_feature_shape[0]
    args.fdim_disease = dataset.disease_feature_shape[0]
    args.rating_vals = dataset.possible_rel_values
    return Net(args=args).to(args.device)


def _prepare_inputs(args, dataset, graph_data):
    return {
        'train_gt_ratings': graph_data['train'][2].to(args.device),
        'train_enc_graph': graph_data['train'][0].int().to(args.device),
        'train_dec_graph': graph_data['train'][1].int().to(args.device),
        'drug_graph': dataset.drug_graph.to(args.device),
        'disease_graph': dataset.disease_graph.to(args.device),
        'drug_sim_feat': th.FloatTensor(dataset.drug_sim_features).to(args.device),
        'disease_sim_feat': th.FloatTensor(dataset.disease_sim_features).to(args.device),
        'drug_feat': dataset.drug_feature.to(args.device),
        'disease_feat': dataset.disease_feature.to(args.device),
    }


def _train_one_epoch(args, model, optimizer, rel_loss, inputs):
    model.train()
    model_output = model(
        inputs['train_enc_graph'],
        inputs['train_dec_graph'],
        inputs['drug_graph'],
        inputs['drug_sim_feat'],
        inputs['drug_feat'],
        inputs['disease_graph'],
        inputs['disease_sim_feat'],
        inputs['disease_feat'],
        False,
    )
    pred_ratings = model_output[0].squeeze(-1)
    contrast_loss = model_output[5] if len(model_output) == 6 else th.tensor(0.0, device=args.device)
    base_loss = rel_loss(pred_ratings, inputs['train_gt_ratings'])
    loss = base_loss + contrast_loss if isinstance(contrast_loss, th.Tensor) else base_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
    optimizer.step()
    return float(loss.item())


def _evaluate(args, model, graph_data, inputs):
    return evaluate(
        args,
        model,
        graph_data,
        inputs['drug_graph'],
        inputs['drug_feat'],
        inputs['drug_sim_feat'],
        inputs['disease_graph'],
        inputs['disease_feat'],
        inputs['disease_sim_feat'],
    )


def _save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def train_fixed_epoch(args, dataset, cv):
    """Train for a fixed budget and evaluate the outer test fold exactly once."""
    fold_dir = os.path.join(args.save_dir, 'fold_{:02d}'.format(cv + 1))
    os.makedirs(fold_dir, exist_ok=True)

    outer_train = dataset.cv_data_dict[cv][0]
    outer_test = dataset.cv_data_dict[cv][1]
    outer_train.to_csv(os.path.join(fold_dir, 'outer_train.csv'), index=False)

    train_graph_data = dataset.build_graph_data(outer_train)
    model = _prepare_model(args, dataset, seed=args.seed + cv)
    inputs = _prepare_inputs(args, dataset, train_graph_data)
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr)
    rel_loss = nn.BCEWithLogitsLoss()
    training_logger = MetricLogger(
        ['epoch', 'train_loss'],
        ['%d', '%.6f'],
        os.path.join(fold_dir, 'training_loss.csv'),
    )

    test_schedule = fixed_epoch_test_schedule(args.train_max_iter)
    print(
        "CV={} phase=fixed_train outer_train={} epochs={} outer_test=hidden".format(
            cv + 1, len(outer_train), args.train_max_iter
        )
    )
    for epoch in range(1, args.train_max_iter + 1):
        train_loss = _train_one_epoch(args, model, optimizer, rel_loss, inputs)
        if epoch % args.train_valid_interval == 0 or epoch == args.train_max_iter:
            training_logger.log(epoch=epoch, train_loss=train_loss)
            print(
                "CV={} phase=fixed_train epoch={}/{} loss={:.6f} "
                "outer_test=hidden".format(
                    cv + 1, epoch, args.train_max_iter, train_loss
                )
            )
    training_logger.close()

    torch.save(
        {
            'model_state_dict': _cpu_state_dict(model),
            'fold': cv + 1,
            'epoch': args.train_max_iter,
            'training_protocol': 'fixed_epoch',
        },
        os.path.join(fold_dir, 'final_checkpoint.pt'),
    )

    if test_schedule != (args.train_max_iter,):
        raise RuntimeError("fixed-epoch protocol must evaluate only at the final epoch")
    outer_test.to_csv(os.path.join(fold_dir, 'outer_test.csv'), index=False)
    test_graph_data = dataset.build_graph_data(outer_train, outer_test)
    test_auroc, test_aupr, y_true, y_score = _evaluate(
        args, model, test_graph_data, inputs
    )
    if len(outer_test) != len(y_true):
        raise RuntimeError("outer test prediction count does not match the split")

    predictions = outer_test[['drug_id', 'disease_id', 'values']].copy().reset_index(drop=True)
    predictions['raw_logit'] = y_score
    predictions.to_csv(os.path.join(fold_dir, 'test_predictions.csv'), index=False)
    _save_json(
        os.path.join(fold_dir, 'test_metrics.json'),
        {
            'fold': cv + 1,
            'test_read_count': 1,
            'fixed_epoch': int(args.train_max_iter),
            'test_auroc': float(test_auroc),
            'test_aupr_trapz': float(test_aupr),
        },
    )
    print(
        "CV={} phase=test fixed_epoch={} test_AUROC={:.6f} "
        "test_AUPR={:.6f}".format(
            cv + 1, args.train_max_iter, test_auroc, test_aupr
        )
    )
    return test_auroc, test_aupr


def train_nested_early_stop(args, dataset, cv):
    """Select epochs on inner validation, refit outer train, then test once."""
    fold_dir = os.path.join(args.save_dir, 'fold_{:02d}'.format(cv + 1))
    os.makedirs(fold_dir, exist_ok=True)

    outer_train = dataset.cv_data_dict[cv][0]
    inner_seed = args.inner_split_seed + cv
    inner_train, inner_validation, split_manifest = stratified_inner_split(
        outer_train,
        validation_ratio=args.inner_val_ratio,
        seed=inner_seed,
    )
    inner_train.to_csv(os.path.join(fold_dir, 'inner_train.csv'), index=False)
    inner_validation.to_csv(os.path.join(fold_dir, 'inner_validation.csv'), index=False)
    split_manifest.update({
        'fold': cv + 1,
        'outer_train_count': int(len(outer_train)),
        'selection_seed': int(args.seed + cv),
        'refit_seed': int(args.seed + args.refit_seed_offset + cv),
    })
    _save_json(os.path.join(fold_dir, 'split_manifest.json'), split_manifest)

    selection_graph_data = dataset.build_graph_data(inner_train, inner_validation)
    selection_model = _prepare_model(args, dataset, seed=args.seed + cv)
    selection_inputs = _prepare_inputs(args, dataset, selection_graph_data)
    selection_optimizer = th.optim.Adam(selection_model.parameters(), lr=args.train_lr)
    rel_loss = nn.BCEWithLogitsLoss()
    tracker = EarlyStoppingTracker(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )
    selection_logger = MetricLogger(
        [
            'epoch', 'train_loss', 'val_auroc', 'val_aupr',
            'best_epoch', 'best_val_aupr', 'patience', 'is_best',
        ],
        ['%d', '%.6f', '%.6f', '%.6f', '%d', '%.6f', '%d', '%d'],
        os.path.join(fold_dir, 'selection_metrics.csv'),
    )
    checkpoint_path = os.path.join(fold_dir, 'best_selection_checkpoint.pt')

    print("CV{} phase=selection inner_train={} validation={} outer_test=hidden".format(
        cv + 1, len(inner_train), len(inner_validation)
    ))
    stopped_epoch = args.train_max_iter
    for epoch in range(1, args.train_max_iter + 1):
        train_loss = _train_one_epoch(
            args, selection_model, selection_optimizer, rel_loss, selection_inputs
        )
        should_validate = epoch % args.selection_interval == 0 or epoch == args.train_max_iter
        if not should_validate:
            continue

        val_auroc, val_aupr, _, _ = _evaluate(
            args, selection_model, selection_graph_data, selection_inputs
        )
        improved = tracker.update(epoch=epoch, aupr=val_aupr, auroc=val_auroc)
        if improved:
            torch.save(
                {
                    'model_state_dict': _cpu_state_dict(selection_model),
                    'fold': cv + 1,
                    'epoch': epoch,
                    'validation_aupr': val_aupr,
                    'validation_auroc': val_auroc,
                    'split_hash': split_manifest['split_hash'],
                },
                checkpoint_path,
            )

        selection_logger.log(
            epoch=epoch,
            train_loss=train_loss,
            val_auroc=val_auroc,
            val_aupr=val_aupr,
            best_epoch=tracker.best_epoch,
            best_val_aupr=tracker.best_aupr,
            patience=tracker.bad_checks,
            is_best=int(improved),
        )
        print(
            "CV={} phase=selection epoch={}/{} loss={:.6f} "
            "val_AUROC={:.6f} val_AUPR={:.6f} best_epoch={} "
            "best_val_AUPR={:.6f} patience={}/{}".format(
                cv + 1,
                epoch,
                args.train_max_iter,
                train_loss,
                val_auroc,
                val_aupr,
                tracker.best_epoch,
                tracker.best_aupr,
                tracker.bad_checks,
                args.early_stop_patience,
            )
        )
        if tracker.should_stop:
            stopped_epoch = epoch
            break
    selection_logger.close()

    if tracker.best_epoch == 0:
        raise RuntimeError("selection phase did not produce a validation checkpoint")
    selection_summary = {
        'fold': cv + 1,
        'primary_metric': 'validation_aupr_trapz',
        'secondary_metric': 'validation_auroc',
        'best_epoch': tracker.best_epoch,
        'best_validation_aupr': tracker.best_aupr,
        'best_validation_auroc': tracker.best_auroc,
        'stopped_epoch': stopped_epoch,
        'validation_interval': args.selection_interval,
        'patience_checks': args.early_stop_patience,
        'min_delta': args.early_stop_min_delta,
    }
    _save_json(os.path.join(fold_dir, 'best_epoch.json'), selection_summary)

    refit_graph_data = dataset.build_graph_data(outer_train)
    refit_model = _prepare_model(
        args,
        dataset,
        seed=args.seed + args.refit_seed_offset + cv,
    )
    refit_inputs = _prepare_inputs(args, dataset, refit_graph_data)
    refit_optimizer = th.optim.Adam(refit_model.parameters(), lr=args.train_lr)
    refit_logger = MetricLogger(
        ['epoch', 'train_loss'],
        ['%d', '%.6f'],
        os.path.join(fold_dir, 'refit_training_loss.csv'),
    )

    print("CV{} phase=refit outer_train={} epochs={} outer_test=hidden".format(
        cv + 1, len(outer_train), tracker.best_epoch
    ))
    for epoch in range(1, tracker.best_epoch + 1):
        train_loss = _train_one_epoch(
            args, refit_model, refit_optimizer, rel_loss, refit_inputs
        )
        if epoch % args.selection_interval == 0 or epoch == tracker.best_epoch:
            refit_logger.log(epoch=epoch, train_loss=train_loss)
            print(
                "CV={} phase=refit epoch={}/{} loss={:.6f}".format(
                    cv + 1, epoch, tracker.best_epoch, train_loss
                )
            )
    refit_logger.close()

    torch.save(
        {
            'model_state_dict': _cpu_state_dict(refit_model),
            'fold': cv + 1,
            'epoch': tracker.best_epoch,
            'split_hash': split_manifest['split_hash'],
        },
        os.path.join(fold_dir, 'refit_final_checkpoint.pt'),
    )

    outer_test = dataset.cv_data_dict[cv][1]
    outer_test.to_csv(os.path.join(fold_dir, 'outer_test.csv'), index=False)
    split_manifest['outer_test_count'] = int(len(outer_test))
    _save_json(os.path.join(fold_dir, 'split_manifest.json'), split_manifest)
    test_graph_data = dataset.build_graph_data(outer_train, outer_test)
    test_auroc, test_aupr, y_true, y_score = _evaluate(
        args, refit_model, test_graph_data, refit_inputs
    )
    if len(outer_test) != len(y_true):
        raise RuntimeError("outer test prediction count does not match the split")
    predictions = outer_test[['drug_id', 'disease_id', 'values']].copy().reset_index(drop=True)
    predictions['raw_logit'] = y_score
    predictions.to_csv(os.path.join(fold_dir, 'test_predictions.csv'), index=False)
    _save_json(
        os.path.join(fold_dir, 'test_metrics.json'),
        {
            'fold': cv + 1,
            'test_read_count': 1,
            'selected_epoch': tracker.best_epoch,
            'test_auroc': float(test_auroc),
            'test_aupr_trapz': float(test_aupr),
        },
    )
    print(
        "CV={} phase=test selected_epoch={} test_AUROC={:.6f} test_AUPR={:.6f}".format(
            cv + 1, tracker.best_epoch, test_auroc, test_aupr
        )
    )
    return test_auroc, test_aupr, tracker.best_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MACH training')

    # 随机种子
    parser.add_argument('--seed', default=125, type=int)

    # 运行设备编号：0 表示 GPU0；-1 表示使用 CPU
    parser.add_argument('--device', default=0, type=int,
                        help='Running device. E.g --device 0, if using cpu, set --device -1')

    # 保存目录（例如模型、log 输出位置）
    parser.add_argument('--save_dir', type=str, help='The saving directory')

    # 保存 ID（便于区分不同实验）
    parser.add_argument('--save_id', type=int, help='The saving log id')

    # 模型非线性激活函数，例如 tanh / relu
    parser.add_argument('--model_activation', type=str, default="tanh")

    # Dropout 比例
    parser.add_argument('--dropout', type=float, default=0.3)

    # GCN 聚合层隐藏单元数
    parser.add_argument('--gcn_agg_units', type=int, default=840)

    # GCN 聚合方式：sum / mean / max
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    # GCN 输出维度
    parser.add_argument('--gcn_out_units', type=int, default=75)

    # 最大训练迭代次数
    parser.add_argument('--train_max_iter', type=int, default=8000)  # 固定训练轮数
    parser.add_argument(
        '--training_protocol',
        type=str,
        default='legacy',
        choices=['legacy', 'fixed_epoch', 'nested_early_stop'],
        help=(
            'legacy keeps the previous loop; fixed_epoch evaluates the outer test '
            'once after the fixed budget; nested_early_stop uses inner validation.'
        ),
    )
    parser.add_argument('--inner_val_ratio', type=float, default=0.1)
    parser.add_argument('--inner_split_seed', type=int, default=2026)
    parser.add_argument('--selection_interval', type=int, default=50)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0001)
    parser.add_argument('--refit_seed_offset', type=int, default=1000)
    parser.add_argument(
        '--resume_fixed_epoch',
        action='store_true',
        help='Reuse fixed-epoch folds only when their audit metrics match.',
    )

    # 梯度裁剪阈值
    parser.add_argument('--train_grad_clip', type=float, default=1.0)

    # 多少次迭代进行一次验证
    parser.add_argument('--train_valid_interval', type=int, default=100)

    # 是否使用对称归一化邻接矩阵
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)

    # MLP / GCN 层隐藏维度 1
    parser.add_argument('--nhid1', type=int, default=500)

    # MLP / GCN 层隐藏维度 2
    parser.add_argument('--nhid2', type=int, default=75)

    # 学习率（适用于 Gdataset）
    parser.add_argument('--train_lr', type=float, default=0.001)

    # 拓扑卷积层数（例如 2 层 Topology-GCN）
    parser.add_argument('--layers', type=int, default=2)

    # 是否共享参数（多层 GCN/TCN 是否共享权重）
    parser.add_argument('--share_param', default=True, action='store_true')

    # 以下参数主要用于融合/对比模块，可保留以备扩展 -----------------------------

    # 双模态注意力。旧参数名 transformer_nhead 仅作为命令兼容别名。
    parser.add_argument(
        '--attention_mode',
        type=str,
        default='modality_tokens',
        choices=['modality_tokens', 'legacy_single_key', 'none'],
        help=(
            'modality_tokens enables real 2x2 topology/similarity attention; '
            'legacy_single_key reproduces the historical degenerate path; '
            'none disables attention for ablation.'
        ),
    )
    parser.add_argument(
        '--attention_heads',
        '--transformer_nhead',
        dest='attention_heads',
        type=int,
        default=3,
        help='Number of modality-attention heads; must divide gcn_out_units.',
    )
    parser.add_argument(
        '--attention_dropout',
        type=float,
        default=0.1,
        help='Dropout inside attention, independent from the global model dropout.',
    )
    fusion_gate_group = parser.add_mutually_exclusive_group()
    fusion_gate_group.add_argument(
        '--use_fusion_gate',
        dest='use_fusion_gate',
        action='store_true',
        help='Use learned topology/similarity gate after attention (default).',
    )
    fusion_gate_group.add_argument(
        '--no_fusion_gate',
        dest='use_fusion_gate',
        action='store_false',
        help='Replace the learned fusion gate with fixed 0.5/0.5 averaging.',
    )
    parser.set_defaults(use_fusion_gate=True)

    # 旧 Transformer 实验类的保留参数；当前 Net 不调用 Transformer。
    parser.add_argument('--transformer_ffn_dim', type=int, default=128)

    # 模态融合方式：并行（parallel）或串行（sequential）
    parser.add_argument('--fusion_mode', type=str, default='parallel', choices=['parallel', 'sequential'])

    # 是否使用对比学习。不要使用 type=bool，字符串 "False" 会被解析为 True。
    contrastive_group = parser.add_mutually_exclusive_group()
    contrastive_group.add_argument(
        '--use_contrastive',
        dest='use_contrastive',
        action='store_true',
    )
    contrastive_group.add_argument(
        '--no_contrastive',
        dest='use_contrastive',
        action='store_false',
    )
    parser.set_defaults(use_contrastive=True)

    # 对比损失权重
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument(
        '--multiscale',
        type=parse_positive_int_tuple,
        default=(2, 4, 8),
        help='Comma-separated MSF reduction ratios, for example: 2,4',
    )
    msf_group = parser.add_mutually_exclusive_group()
    msf_group.add_argument('--use_msf', dest='use_msf', action='store_true')
    msf_group.add_argument('--no_msf', dest='use_msf', action='store_false')
    parser.set_defaults(use_msf=True)

    feature_branch_group = parser.add_mutually_exclusive_group()
    feature_branch_group.add_argument(
        '--use_feature_branch', dest='use_feature_branch', action='store_true'
    )
    feature_branch_group.add_argument(
        '--no_feature_branch', dest='use_feature_branch', action='store_false'
    )
    parser.set_defaults(use_feature_branch=True)

    topology_branch_group = parser.add_mutually_exclusive_group()
    topology_branch_group.add_argument(
        '--use_topology_branch', dest='use_topology_branch', action='store_true'
    )
    topology_branch_group.add_argument(
        '--no_topology_branch', dest='use_topology_branch', action='store_false'
    )
    parser.set_defaults(use_topology_branch=True)

    # Focal Loss 参数 gamma
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # 对比损失中的 margin 值
    parser.add_argument('--margin', type=float, default=0.3)

    # ---------------- Gdataset 最优参数相关 ----------------------------------

    # 数据集名称，如 "lrssl"
    parser.add_argument('--data_name', default='Gdataset', type=str)

    # 每个节点采样的邻居数量
    parser.add_argument('--num_neighbor', type=int, default=4)

    args = parser.parse_args()
    if not args.use_feature_branch and not args.use_topology_branch:
        parser.error('at least one of feature_branch/topology_branch must be enabled')
    if not args.use_feature_branch or not args.use_topology_branch:
        args.attention_mode = 'none'
        args.use_fusion_gate = False
        args.use_contrastive = False
        args.use_msf = False
    print(args)

    args.device = th.device(args.device) if args.device >= 0 else th.device('cpu')
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    # ========== 修改：收集所有折的结果 ==========
    all_aurocs = []  # 存储所有折的最佳 AUROC
    all_auprs = []  # 存储所有折的最佳 AUPR
    selected_epochs = []

    # One 10-fold cross-validation run; the historical loop has one iteration.
    for times in range(1, 2):
        print("++++++++++++++++++times", str(times), "++++++++++++++++++++++")

        args.save_dir = args.data_name + "_" + ''.join(str(times) + 'time')
        args.save_dir = os.path.join("neighbor_num1", args.save_dir)
        if args.training_protocol != 'legacy':
            args.save_dir = os.path.join(args.save_dir, args.training_protocol)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        _save_json(
            os.path.join(args.save_dir, 'protocol_config.json'),
            {
                key: str(value) if key == 'device' else value
                for key, value in vars(args).items()
                if key != 'save_dir'
            },
        )

        dataset = DrugDataLoader(
            args.data_name, args.device,
            symm=args.gcn_agg_norm_symm,
            k=args.num_neighbor,
            build_legacy_graphs=args.training_protocol == 'legacy',
        )
        print("Loading dataset finished ...\n")

        for cv in range(0, 10):
            args.save_id = cv + 1
            print("===============" + str(cv + 1) + "=================")
            if args.training_protocol == 'legacy':
                graph_data = dataset.data_cv[cv]
                best_auroc, best_aupr = train(args, dataset, graph_data, cv)
            elif args.training_protocol == 'nested_early_stop':
                best_auroc, best_aupr, selected_epoch = train_nested_early_stop(
                    args, dataset, cv
                )
                selected_epochs.append(selected_epoch)
            else:
                completed_metrics = None
                if args.resume_fixed_epoch:
                    completed_metrics = load_completed_fixed_fold(
                        os.path.join(
                            args.save_dir,
                            'fold_{:02d}'.format(cv + 1),
                            'test_metrics.json',
                        ),
                        expected_fold=cv + 1,
                        expected_epoch=args.train_max_iter,
                    )
                if completed_metrics is None:
                    best_auroc, best_aupr = train_fixed_epoch(args, dataset, cv)
                else:
                    best_auroc, best_aupr = completed_metrics
                    print(
                        "CV={} phase=resume completed_fold_reused "
                        "fixed_epoch={} test_AUROC={:.6f} test_AUPR={:.6f}".format(
                            cv + 1,
                            args.train_max_iter,
                            best_auroc,
                            best_aupr,
                        )
                    )

            # 收集每一折的结果
            all_aurocs.append(best_auroc)
            all_auprs.append(best_aupr)

    # ========== 修改：计算并输出 10 折交叉验证的均值和标准差 ==========
    print("\n" + "=" * 60)
    print("10-Fold Cross Validation Results Summary")
    print("Training protocol: {}".format(args.training_protocol))
    print("=" * 60)

    # 打印每一折的结果
    print("\nResults for each fold:")
    print("-" * 40)
    for i, (auc, apr) in enumerate(zip(all_aurocs, all_auprs)):
        epoch_text = (
            f", selected epoch = {selected_epochs[i]}"
            if args.training_protocol == 'nested_early_stop'
            else ""
        )
        print(f"  Fold {i + 1}: AUROC = {auc:.4f}, AUPR = {apr:.4f}{epoch_text}")

    # 转换为 numpy 数组计算统计量
    aurocs_array = np.array(all_aurocs)
    auprs_array = np.array(all_auprs)

    # 计算均值和标准差
    auroc_mean = np.mean(aurocs_array)
    auroc_std = np.std(aurocs_array)
    aupr_mean = np.mean(auprs_array)
    aupr_std = np.std(auprs_array)

    # 输出最终结果
    print("\n" + "-" * 40)
    print("Final Results:")
    print("-" * 40)
    print(f"  AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"  AUPR:  {aupr_mean:.4f} ± {aupr_std:.4f}")
    print("=" * 60)

    # ========== 保存结果到文件 ==========
    results_file = os.path.join(args.save_dir, 'final_results.txt')
    with open(results_file, 'w') as f:
        f.write("10-Fold Cross Validation Results\n")
        f.write("Training protocol: {}\n".format(args.training_protocol))
        f.write("=" * 50 + "\n\n")

        f.write("Results for each fold:\n")
        f.write("-" * 40 + "\n")
        for i, (auc, apr) in enumerate(zip(all_aurocs, all_auprs)):
            epoch_text = (
                f", selected epoch = {selected_epochs[i]}"
                if args.training_protocol == 'nested_early_stop'
                else ""
            )
            f.write(f"  Fold {i + 1}: AUROC = {auc:.4f}, AUPR = {apr:.4f}{epoch_text}\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("Final Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}\n")
        f.write(f"  AUPR:  {aupr_mean:.4f} ± {aupr_std:.4f}\n")
        f.write("=" * 50 + "\n")

    print(f"\nResults saved to: {results_file}")
