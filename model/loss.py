import torch
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # 将预测结果应用 sigmoid 函数


        # 展平张量
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        # 计算交集和并集
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        # 返回 1 - Dice 系数作为损失
        return 1 - dice

class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class FixedLoss(nn.Module):
    def __init__(self,alpha=0.2,beta=0.8):
        super(FixedLoss, self).__init__()
        self.alpha=alpha
        self.beta=beta
        self.dice= DiceLoss()
        self.ce = nn.BCELoss()
        #self.focal=BinaryFocalLossWithLogits()
    def forward(self, preds, targets):
        # 将预测结果应用 sigmoid 函数
        preds = torch.sigmoid(preds)
        return self.beta*self.dice(preds,targets)+self.alpha*self.ce(preds,targets),self.dice(preds,targets),self.ce(preds,targets)

    # def forward(self, preds, targets):
    #     dice_loss = self.dice(torch.sigmoid(preds), targets)
    #     focal_loss = self.focal(preds, targets)
    #     total = self.beta * dice_loss + self.alpha * focal_loss
    #     return total, dice_loss, focal_loss

class RandConvLoss(nn.Module):
    ##########syl加，以前没有
    def __init__(self):
        super().__init__()

        self.segloss = FixedLoss()
        self.eps = 1e-6
    ###############
    def _safe_prob(self, x):
        return torch.clamp(x, self.eps, 1. - self.eps)

    def forward(self, out1, mask, out2=None, out3=None):
        seg_loss = self.segloss(out1, mask)

        if out2 is None or out3 is None:
            return seg_loss[0]

        # sigmoid → prob
        p1 = self._safe_prob(torch.sigmoid(out1))
        p2 = self._safe_prob(torch.sigmoid(out2))
        p3 = self._safe_prob(torch.sigmoid(out3))

        # 如果是全背景 slice，直接跳过 consistency
        if mask.sum() == 0:
            inv_loss = torch.tensor(0., device=out1.device)
        else:
            # mixture 用 stop-grad，防止互相拖死
            p_mix = ((p1 + p2 + p3) / 3.).detach()
            p_mix = self._safe_prob(p_mix)

            log_p_mix = torch.log(p_mix)

            inv_loss = (
                               F.kl_div(log_p_mix, p1, reduction='batchmean') +
                               F.kl_div(log_p_mix, p2, reduction='batchmean') +
                               F.kl_div(log_p_mix, p3, reduction='batchmean')
                       ) / 3.

        total_loss = seg_loss[0] + 0.0002 * inv_loss

        return total_loss, seg_loss[1], seg_loss[2], 0.0002 * inv_loss
    # def forward(self, out1,mask,out2=None,out3=None):
    #
    #     if out2 is not None and out3 is not None:
    #         p_clean, p_aug1, p_aug2 = F.sigmoid(
    #             out1), F.sigmoid(
    #             out2), F.sigmoid(
    #             out3)
    #
    #         p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    #         inv_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
    #                      F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
    #                      F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    #         seg_loss=self.segloss(out1,mask)
    #         return seg_loss[0]+0.0002*inv_loss,seg_loss[1],seg_loss[2],0.0002*inv_loss
    #     else:
    #         return self.segloss(out1,mask)[0],

class ASDGKLLoss(nn.Module):
    def __init__(self):
        super(ASDGKLLoss, self).__init__()

    def forward(self, prob1,prob2):
        prob1 = F.sigmoid(prob1)
        prob2 = F.sigmoid(prob2)
        return (nn.KLDivLoss()(prob1.log(),prob2)+nn.KLDivLoss()(prob2.log(),prob1))/2

class ASDGPatchNCELoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        # self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.mask_dtype=torch.bool
    def forward(self, feat_q, feat_k, batch_size):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
class ASDGLoss1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.seg_loss=FixedLoss()
        self.kl_loss=ASDGKLLoss()
    def forward(self, predict1, predict2,label):
        return (self.seg_loss(predict1,label)[0] +self.seg_loss(predict2,label)[0])/2 + self.kl_loss(predict1,predict2),
class ASDGLoss2(nn.Module):
    def __init__(self,):
        super().__init__()
        self.kl_loss=ASDGKLLoss()
    def forward(self, new_predict1, new_predict2,miloss):
        return -self.kl_loss(new_predict1,new_predict2)+miloss,

def ASDGMI_loss(src, tgt, nets):

    nce_layers = 2

    crit  = ASDGPatchNCELoss().cuda()
    netG, netF = nets

    n_layers = len(nce_layers)
    feat_q = netG(tgt, nce_layers, encode_only=True)

    feat_k = netG(src, nce_layers, encode_only=True)
    feat_k_pool, sample_ids = netF(feat_k, 256, None)
    feat_q_pool, _ = netF(feat_q, 256, sample_ids)

    bs = src.shape[0]

    total_nce_loss = 0.0
    for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, nce_layers):
        loss = crit(f_q, f_k, bs) * 1.0
        total_nce_loss += loss.mean()

    return total_nce_loss / n_layers

if __name__=='__main__':
    d1=torch.ones((1,3,256,256))
    d2=torch.ones((1,3,256,256))
    l=RandConvLoss()

    print(l(d1,d1,d1,d2))