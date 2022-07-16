# Contrastive loss implementation
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

def contrast(feats, labels, imnet_mask=None, sim_weight=None, feat_ext=None):
    contrast_count = feats.shape[1]
    contrast_feature = torch.cat(torch.unbind(feats, dim=1), dim=0)
    if feat_ext is not None:
        contrast_feature_ext = torch.cat(torch.unbind(feat_ext, dim=1), dim=0)
        anchor_dot_contrast = torch.matmul(contrast_feature,
                              torch.transpose(contrast_feature_ext, 0, 1))
    else:
        anchor_dot_contrast = torch.matmul(contrast_feature,
                              torch.transpose(contrast_feature, 0, 1))
    # Numerical stablization
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    
    mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float().cuda()
    neg_mask = 1 - mask

    # Eliminate self contrast
    logits_mask = torch.eye(mask.shape[0],
                            mask.shape[0]).logical_not().to(mask.device)
    mask = mask * logits_mask

    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    exp_logits = torch.exp(logits)

    log_prob = logits - torch.log(exp_logits + neg_logits)

    if imnet_mask is not None:
        # Don't compute similarity with thing-class features
        mask = mask * imnet_mask

    if sim_weight is not None:
        if feat_ext is None:
            log_prob = log_prob * sim_weight
        else:
            log_prob = log_prob * (1 - sim_weight)
    return log_prob * mask, mask

def multi_resolution_cl(lr_feats, hr_feats, labels, att,
                        hr_lbl, hr_max_feats=None, hr_max_lbl=None,
                        pseudo_weight=None, hr_pseudo_weight=None,
                        fdist_classes=None):
    labels = labels.view(labels.shape[0], -1, 1)
    labels = labels.view(-1, 1)
    
    hr_lbl = hr_lbl.view(hr_lbl.shape[0], -1, 1)
    hr_lbl = hr_lbl.view(-1, 1)
    if fdist_classes is not None:
        fdclasses = torch.tensor(fdist_classes, device=labels.device)
        fdist_mask = torch.any(labels[..., None] == fdclasses, -1)
        imnet_mask = fdist_mask.logical_not().view(-1, 1)
        imnet_mask = torch.eq(imnet_mask, imnet_mask.transpose(0, 1)).float()
    
    # Predicted label confidence similarity weights
    if pseudo_weight is not None:
        pw = pseudo_weight.view(-1, 1)
        lbl_sim_weight = torch.matmul(pw, pw.transpose(0, 1))
    
    lr_feats = lr_feats.permute(0,2,3,1)
    lr_feats = lr_feats.view(lr_feats.shape[0], -1, lr_feats.shape[3])
    lr_feats = F.normalize(lr_feats, dim = -1)
    hr_feats = hr_feats.permute(0,2,3,1)
    hr_feats = hr_feats.view(hr_feats.shape[0], -1, hr_feats.shape[3])
    hr_feats = F.normalize(hr_feats, dim = -1)

    mix_mask = torch.where(hr_lbl == 0, 0, 1)
    mix_lbl = mix_mask * hr_lbl + (1 - mix_mask) * labels

    att = att.view(labels.shape[0], -1, 1)
    att = att.view(-1, 1)
    # Learned similarity weight
    ld_sim_weight = torch.matmul(att, att.transpose(0, 1))
    
    ld_sim_weight = ld_sim_weight * mix_mask.view(-1, 1)
    lr_cl, mask = contrast(lr_feats, labels, imnet_mask=imnet_mask,
                           sim_weight=ld_sim_weight)
    mx_cl, _ = contrast(lr_feats, mix_lbl, imnet_mask=imnet_mask,
                        sim_weight=ld_sim_weight, feat_ext=hr_feats)
    log_prob = lr_cl + mx_cl
    if pseudo_weight is not None:
        log_prob = log_prob * lbl_sim_weight

    # It may happen that only one pixel belonging to a class is present in
    # the reduced gt map.
    den = mask.sum(1)
    den[den == 0] = 1
    mean_log_prob_pos = -log_prob.sum(1) / den
    loss = 0.5 * mean_log_prob_pos.mean()
    return loss

def contrastive_loss(feats_, labels_, feats_teacher_=None, fdist_classes=None,
                     pseudo_weight=None, temperature=1, base_temperature=1):    
    feats_ = feats_.permute(0,2,3,1)
    feats_ = feats_.view(feats_.shape[0], -1, feats_.shape[3])
    feats_ = F.normalize(feats_, dim = -1) 
    labels_ = labels_.view(labels_.shape[0], -1, 1)
    labels = labels_.view(-1, 1)

    # Eliminate all thing-class features
    if fdist_classes is not None:
        fdclasses = torch.tensor(fdist_classes, device=labels_.device)
        fdist_mask = torch.any(labels_[..., None] == fdclasses, -1)
        imnet_mask = fdist_mask.logical_not().view(-1, 1)
        imnet_mask = torch.eq(imnet_mask, imnet_mask.transpose(0, 1)).float()
    
    if pseudo_weight is not None:
        pw = pseudo_weight.view(-1, 1)
        sim_weight = torch.matmul(pw, pw.transpose(0, 1))

    mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float().cuda()

    contrast_count = feats_.shape[1]
    contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

    anchor_dot_contrast = torch.matmul(contrast_feature,
                                       torch.transpose(contrast_feature, 0, 1))
      
    # Numerical stablization
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    neg_mask = 1 - mask

    # Eliminate self contrast
    logits_mask = torch.eye(mask.shape[0], mask.shape[0]).logical_not().to(mask.device)
    mask = mask * logits_mask

    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    exp_logits = torch.exp(logits)

    log_prob = logits - torch.log(exp_logits + neg_logits)

    if fdist_classes is not None:
        # Don't compute similarity with thing-class features
        mask = mask * imnet_mask
    
    den = mask.sum(1)
    den[den == 0] = 1

    if pseudo_weight is not None:
        log_prob = log_prob * sim_weight

    mean_log_prob_pos = (mask * log_prob).sum(1) / den
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()
    return loss

