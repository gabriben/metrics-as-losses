#https://github.com/Alibaba-MIIL/ASL/blob/7114637713619f01906ed73bbfa182565cd3a77d/src/helper_functions/helper_functions.py#L49

import numpy as np

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

#def TFmAP()

# def validate_multi(val_loader, model, ema_model):
#     print("starting validation")
#     Sig = torch.nn.Sigmoid()
#     preds_regular = []
#     preds_ema = []
#     targets = []
#     for i, (input, target) in enumerate(val_loader):
#         target = target
#         target = target.max(dim=1)[0]
#         # compute output
#         with torch.no_grad():
#             with autocast():
#                 output_regular = Sig(model(input.cuda())).cpu()
#                 output_ema = Sig(ema_model.module(input.cuda())).cpu()

#         # for mAP calculation
#         preds_regular.append(output_regular.cpu().detach())
#         preds_ema.append(output_ema.cpu().detach())
#         targets.append(target.cpu().detach())

#     mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
#     mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
#     print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
#     return max(mAP_score_regular, mAP_score_ema)
