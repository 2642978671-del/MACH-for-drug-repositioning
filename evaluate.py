import torch as th
from sklearn import metrics


def evaluate(args, model, graph_data,
             drug_graph, drug_feat, drug_sim_feat,
             dis_graph, dis_feat, dis_sim_feat):
    rating_values = graph_data['test'][2]
    enc_graph = graph_data['test'][0].int().to(args.device)
    dec_graph = graph_data['test'][1].int().to(args.device)

    model.eval()
    with th.no_grad():
        Two_Stage = False  # ← 添加这一行！

        pred_ratings, _, _, _, _, _ = model(enc_graph, dec_graph,
                                            drug_graph, drug_sim_feat, drug_feat,
                                            dis_graph, dis_sim_feat, dis_feat,
                                            Two_Stage)

    y_score = pred_ratings.view(-1).cpu().tolist()
    y_true = rating_values.cpu().tolist()

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)

    return auc, aupr, y_true, y_score
