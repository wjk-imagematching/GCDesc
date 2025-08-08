import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def dual_softmax_loss(X, Y, weights=None, temp=0.2):
    """
    Dual-softmax loss with optional per-pair weighting.
    Args:
        X, Y (torch.Tensor): Descriptors to match, shape (N, C).
        weights (torch.Tensor, optional): Per-pair weights, shape (N,).
    """
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp

    log_p12 = F.log_softmax(dist_mat, dim=1)
    log_p21 = F.log_softmax(dist_mat.t(), dim=1)

    target = torch.arange(len(X), device=X.device)

    # 计算逐对的损失
    loss12_per_pair = F.nll_loss(log_p12, target, reduction='none')  # (N,)
    loss21_per_pair = F.nll_loss(log_p21, target, reduction='none')  # (N,)

    loss_per_pair = (loss12_per_pair + loss21_per_pair) / 2.0

    # 应用权重
    if weights is not None:
        # 加权平均: sum(loss * weight) / sum(weight)
        # 这确保了损失的梯度大小与权重成正比
        sum_of_weights = weights.sum()
        if sum_of_weights > 1e-8:
            final_loss = (loss_per_pair * weights).sum() / sum_of_weights
        else:
            final_loss = loss_per_pair.mean()  # Fallback
    else:
        final_loss = loss_per_pair.mean()

    # Confidence calculation remains the same
    with torch.no_grad():
        conf = torch.exp(log_p12).max(dim=-1)[0] * torch.exp(log_p21).max(dim=-1)[0]

    return final_loss, conf

class LiftFeatLoss(nn.Module):
    def __init__(self,device,lam_descs=1,lam_fb_descs=1,lam_kpts=1,lam_heatmap=1,lam_normals=0.5,lam_coordinates=1,lam_fb_coordinates=1,depth_spvs=False):
        super().__init__()

        # loss parameters
        self.lam_descs=lam_descs
        self.lam_fb_descs=lam_fb_descs
        self.lam_kpts=lam_kpts
        self.lam_heatmap=lam_heatmap
        self.lam_normals=lam_normals
        self.lam_coordinates=lam_coordinates
        self.lam_fb_coordinates=lam_fb_coordinates
        self.depth_spvs=depth_spvs
        self.running_descs_loss=0
        self.running_kpts_loss=0
        self.running_heatmaps_loss=0
        self.loss_descs=0
        self.loss_fb_descs=0
        self.loss_kpts=0
        self.loss_heatmaps=0
        self.loss_normals=0
        self.loss_coordinates=0
        self.loss_fb_coordinates=0
        self.acc_coarse=0
        self.acc_fb_coarse=0
        self.acc_kpt=0
        self.acc_coordinates=0
        self.acc_fb_coordinates=0

        self.kpt_base_weight = 1
        self.kpt_alpha_curvature = 2
        self.desc_min_weight = 0.1

        # device
        self.dev=device

        self.sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                           dtype=torch.float32, device=self.dev).unsqueeze(0).unsqueeze(0)
        self.sobel_y_kernel = self.sobel_x_kernel.transpose(2, 3)

    def check_accuracy(self,m1,m2,pts1=None,pts2=None,plot=False):
        with torch.no_grad():
            #dist_mat = torch.cdist(X,Y)
            dist_mat = m1 @ m2.t()
            nn = torch.argmax(dist_mat, dim=1)
            #nn = torch.argmin(dist_mat, dim=1)
            correct = nn == torch.arange(len(m1), device = m1.device)

            # if pts1 is not None and plot:
            #     import matplotlib.pyplot as plt
            #     canvas = torch.zeros((60, 80),device=m1.device)
            #     pts1 = pts1[~correct]
            #     canvas[pts1[:,1].long(), pts1[:,0].long()] = 1
            #     canvas = canvas.cpu().numpy()
            #     plt.imshow(canvas), plt.show()
            acc = correct.sum().item() / len(m1)
            return acc

        # --- 权重计算辅助函数 ---
    def _compute_weights_from_teacher_normal(self, teacher_normal_chw):
        with torch.no_grad():
            normal_map_bchw = teacher_normal_chw.unsqueeze(0)
            # 使用 'replicate' padding 代替 'same' 以兼容旧版PyTorch
            grad_nx_x = F.conv2d(normal_map_bchw[:, 0:1, :, :], self.sobel_x_kernel, padding=1)
            grad_nx_y = F.conv2d(normal_map_bchw[:, 0:1, :, :], self.sobel_y_kernel, padding=1)
            grad_ny_x = F.conv2d(normal_map_bchw[:, 1:2, :, :], self.sobel_x_kernel, padding=1)
            grad_ny_y = F.conv2d(normal_map_bchw[:, 1:2, :, :], self.sobel_y_kernel, padding=1)
            grad_nz_x = F.conv2d(normal_map_bchw[:, 2:3, :, :], self.sobel_x_kernel, padding=1)
            grad_nz_y = F.conv2d(normal_map_bchw[:, 2:3, :, :], self.sobel_y_kernel, padding=1)
            curvature_map_sq = grad_nx_x ** 2 + grad_ny_x ** 2 + grad_nz_x ** 2 + grad_nx_y ** 2 + grad_ny_y ** 2 + grad_nz_y ** 2
            curvature_map = torch.sqrt(curvature_map_sq).squeeze(0).squeeze(0)
            min_val, max_val = curvature_map.min(), curvature_map.max()
            range_val = max_val - min_val
            c_norm = (curvature_map - min_val) / (range_val + 1e-7) if range_val > 1e-7 else torch.zeros_like(
                curvature_map)

            # 正向权重 (for keypoints and descriptors)
            W_keypoint = self.kpt_base_weight + self.kpt_alpha_curvature * c_norm
            # 反向权重 (for normals)
            W_smoothness = self.desc_min_weight + (1.0 - self.desc_min_weight) * (1.0 - c_norm)
            W_keypoint_s8 = F.avg_pool2d(W_keypoint.unsqueeze(0).unsqueeze(0), kernel_size=8).squeeze()

        return W_keypoint_s8, W_smoothness

    # 在 LiftFeatLoss 类中

    def compute_descriptors_loss(self, descs1, descs2, pts_s8_list, keypoint_weights_s8_list1=None,
                                 keypoint_weights_s8_list2=None):
        calculated_losses, conf_list = [], []
        total_acc, valid_batches = 0, 0

        B = len(keypoint_weights_s8_list1)
        # B = len(pts_s8_list)

        for b in range(B):
            if pts_s8_list[b].shape[0] == 0:
                conf_list.append(None)
                continue

            pts1_s8, pts2_s8 = pts_s8_list[b][:, :2], pts_s8_list[b][:, 2:]

            m1 = descs1[b, :, pts1_s8[:, 1].long(), pts1_s8[:, 0].long()].permute(1, 0)
            m2 = descs2[b, :, pts2_s8[:, 1].long(), pts2_s8[:, 0].long()].permute(1, 0)

            if m1.shape[0] == 0:
                conf_list.append(None)
                continue

            pair_weights = None
            if keypoint_weights_s8_list1 is not None and keypoint_weights_s8_list2 is not None:
                weights1_at_pts = keypoint_weights_s8_list1[b][pts1_s8[:, 1].long(), pts1_s8[:, 0].long()]
                weights2_at_pts = keypoint_weights_s8_list2[b][pts2_s8[:, 1].long(), pts2_s8[:, 0].long()]
                pair_weights = (weights1_at_pts + weights2_at_pts) / 2.0  # (N_matches,)

            # 调用新的加权损失函数
            loss_per, conf_per = dual_softmax_loss(m1, m2, weights=pair_weights)

            calculated_losses.append(loss_per.unsqueeze(0))
            conf_list.append(conf_per)
            total_acc += self.check_accuracy(m1, m2)
            valid_batches += 1

        if not calculated_losses:
            return torch.tensor(0.0, device=self.dev), 0.0, []

        final_loss = torch.cat(calculated_losses).mean()
        final_acc = total_acc / max(1, valid_batches)
        return final_loss, final_acc, conf_list

    def alike_distill_loss(self, kpts, alike_kpts, keypoint_weight_map_s8_hw=None):
        """
        Computes the keypoint distillation loss, optionally weighted by a positive curvature map.
        Args:
            kpts (torch.Tensor): Predicted keypoint logits map (C, H_s8, W_s8), C=65.
            alike_kpts (torch.Tensor): Ground truth keypoint coordinates (N_gt, 2) in full resolution.
            keypoint_weight_map_s8_hw (torch.Tensor, optional): Positive curvature weights at 1/8 resolution (H_s8, W_s8).
        """
        C, H_s8, W_s8 = kpts.shape

        # --- 1. 生成GT标签 (逻辑不变) ---
        kpts_permuted = kpts.permute(1, 2, 0)
        with torch.no_grad():
            labels = torch.ones((H_s8, W_s8), dtype=torch.long, device=kpts.device) * 64
            if alike_kpts.shape[0] > 0:
                kpts_s8_coords = (alike_kpts / 8).long()
                offsets = ((alike_kpts / 8) - kpts_s8_coords).mul(8).long()
                linear_offsets = offsets[:, 0] + 8 * offsets[:, 1]
                kpt_y_s8, kpt_x_s8 = torch.clamp(kpts_s8_coords[:, 1], 0, H_s8 - 1), torch.clamp(kpts_s8_coords[:, 0],
                                                                                                 0, W_s8 - 1)
                labels[kpt_y_s8, kpt_x_s8] = linear_offsets

        # --- 2. 展平并采样正负样本 (逻辑不变) ---
        kpts_flat, labels_flat = kpts_permuted.reshape(-1, C), labels.reshape(-1)
        mask_pos = labels_flat < 64
        idxs_pos = torch.where(mask_pos)[0]
        if idxs_pos.numel() == 0:
            return torch.tensor(0.0, device=kpts.device), 0.0
        idxs_neg = torch.where(~mask_pos)[0]
        num_neg_samples = min(len(idxs_pos) // 32, idxs_neg.numel())
        if num_neg_samples > 0:
            perm = torch.randperm(idxs_neg.size(0), device=kpts.device)[:num_neg_samples]
            idxs_sampled = torch.cat([idxs_pos, idxs_neg[perm]])
        else:
            idxs_sampled = idxs_pos
        kpts_sampled, labels_sampled = kpts_flat[idxs_sampled], labels_flat[idxs_sampled]

        # --- 3. 计算准确率 (逻辑不变) ---
        with torch.no_grad():
            acc = (kpts_sampled.max(dim=-1)[
                       1] == labels_sampled).float().mean().item() if kpts_sampled.numel() > 0 else 0.0

        # --- 4. 计算加权损失 (核心修改) ---
        kpts_log_softmax = F.log_softmax(kpts_sampled, dim=-1)

        # 计算逐样本的损失
        pointwise_loss = F.nll_loss(kpts_log_softmax, labels_sampled, reduction='none')

        if keypoint_weight_map_s8_hw is not None:
            # 如果提供了权重图，则应用权重

            # 将 1/8 分辨率的权重图展平
            weights_flat = keypoint_weight_map_s8_hw.reshape(-1)

            # 直接使用采样点的索引 `idxs_sampled` 从展平的权重图中获取权重
            weights_sampled = weights_flat[idxs_sampled]

            # 计算加权平均损失
            sum_of_weights = weights_sampled.sum()
            if sum_of_weights > 1e-8:
                final_loss = (pointwise_loss * weights_sampled).sum() / sum_of_weights
            else:
                final_loss = pointwise_loss.mean()  # Fallback
        else:
            # 如果没有提供权重，则计算普通平均损失
            final_loss = pointwise_loss.mean()

        return final_loss, acc

    def compute_keypoints_loss(self, kpts1, kpts2, alike_kpts1, alike_kpts2, keypoint_weights_s8_list1=None,
                               keypoint_weights_s8_list2=None):
        """
        Computes keypoint distillation loss, passing the pre-computed weights to the sub-function.
        """
        calculated_losses, total_acc, valid_batches = [], 0, 0
        B = len(keypoint_weights_s8_list1)
        # B = kpts1.shape[0]

        for b in range(B):
            # 获取当前batch item对应的权重图 (已经是1/8分辨率)
            weight1 = keypoint_weights_s8_list1[b] if keypoint_weights_s8_list1 is not None else None
            weight2 = keypoint_weights_s8_list2[b] if keypoint_weights_s8_list2 is not None else None

            # 调用修改后的 alike_distill_loss
            loss_per1, acc_per1 = self.alike_distill_loss(kpts1[b], alike_kpts1[b], weight1)
            loss_per2, acc_per2 = self.alike_distill_loss(kpts2[b], alike_kpts2[b], weight2)

            if not (torch.isnan(loss_per1) or torch.isnan(loss_per2)):
                loss_per_pair = (loss_per1 + loss_per2) / 2.0
                acc_per_pair = (acc_per1 + acc_per2) / 2.0
                calculated_losses.append(loss_per_pair.unsqueeze(0))
                total_acc += acc_per_pair
                valid_batches += 1

        if not calculated_losses:
            return torch.tensor(0.0, device=self.dev), 0.0

        final_loss = torch.cat(calculated_losses).mean()
        final_acc = total_acc / max(1, valid_batches)

        return final_loss, final_acc

    # normal_loss 方法与上一个回答中的版本类似，它接收权重图
    def normal_loss(self, pred_normal_chw, target_normal_chw, smoothness_weight_map_hw=None):
        if pred_normal_chw.shape[-2:] != target_normal_chw.shape[-2:]:
            target_h, target_w = target_normal_chw.shape[-2:]
            pred_normal_chw_aligned = F.interpolate(pred_normal_chw.unsqueeze(0),
                                                    size=(target_h, target_w),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0)
            pred_normal_chw_aligned = F.normalize(pred_normal_chw_aligned, p=2, dim=0)
        else:
            pred_normal_chw_aligned = pred_normal_chw

        # print(f"[Debug normal_loss] pred_aligned shape: {pred_normal_chw_aligned.shape}")  # 应该是 (3, H_gt, W_gt)
        # print(f"[Debug normal_loss] target_normal shape: {target_normal_chw.shape}")  # 应该是 (3, H_gt, W_gt)

        dot = torch.cosine_similarity(pred_normal_chw_aligned, target_normal_chw, dim=0)
        dot_clamped = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        pointwise_angular_loss = torch.acos(dot_clamped)  # (H_gt, W_gt)

        if smoothness_weight_map_hw is not None:
            if smoothness_weight_map_hw.shape[-2:] != pointwise_angular_loss.shape[-2:]:
                smoothness_weight_map_hw_aligned = F.interpolate(
                    smoothness_weight_map_hw.unsqueeze(0).unsqueeze(0),  # (1,1,H,W)
                    size=pointwise_angular_loss.shape[-2:],
                    mode='nearest'  # 权重图用最近邻插值可能更好
                ).squeeze(0).squeeze(0)
            else:
                smoothness_weight_map_hw_aligned = smoothness_weight_map_hw

            weighted_loss_sum = torch.sum(pointwise_angular_loss * smoothness_weight_map_hw_aligned)
            sum_of_weights = smoothness_weight_map_hw_aligned.sum()

            if sum_of_weights > 1e-6:
                loss = weighted_loss_sum / sum_of_weights
            else:
                loss = torch.mean(pointwise_angular_loss)  # Fallback
                # print("[Normal Loss Warning] Sum of smoothness weights is near zero. Using unweighted mean.")
        else:
            loss = torch.mean(pointwise_angular_loss)

        if torch.isnan(loss):
            # print("[Normal Loss Warning] NaN detected.")
            return torch.tensor(0.0, device=self.dev, requires_grad=pred_normal_chw.requires_grad)
        return loss

    # 修改 compute_normals_loss 以接收外部权重
    def compute_normals_loss(self,
                             normals1_batch, normals2_batch,  # 预测的法线图
                             DA_normals1_list, DA_normals2_list,  # GT法线图列表
                             smoothness_weights1_list=None, smoothness_weights2_list=None,  # <--- 接收外部权重
                             megadepth_batch_size=None, coco_batch_size=None):

        calculated_losses = []


        num_da_samples = len(DA_normals1_list)

        # 检查传入的权重列表是否有效
        use_weights = (smoothness_weights1_list is not None and
                       smoothness_weights2_list is not None and
                       len(smoothness_weights1_list) == num_da_samples and
                       len(smoothness_weights2_list) == num_da_samples)

        for b in range(num_da_samples):
            # if b >= pred_normals1_megadepth.shape[0]: continue

            pred_n1_chw = normals1_batch[b]  # (3, H_pred, W_pred)
            pred_n2_chw = normals2_batch[b]

            teacher_n1_chw = DA_normals1_list[b].to(self.dev)
            teacher_n2_chw = DA_normals2_list[b].to(self.dev)

            # 获取对应的权重图
            weights1_hw = smoothness_weights1_list[b].to(self.dev) if use_weights else None
            weights2_hw = smoothness_weights2_list[b].to(self.dev) if use_weights else None

            # print(pred_n1_chw.shape)
            # print(teacher_n1_chw.shape)
            # 调用 self.normal_loss，传入权重 (如果存在)
            loss_per1 = self.normal_loss(pred_n1_chw, teacher_n1_chw, weights1_hw)
            loss_per2 = self.normal_loss(pred_n2_chw, teacher_n2_chw, weights2_hw)

            if not (torch.isnan(loss_per1) or torch.isnan(loss_per2)):
                loss_per_pair = (loss_per1 + loss_per2) / 2.0
                calculated_losses.append(loss_per_pair.unsqueeze(0))

        if calculated_losses:
            return torch.cat(calculated_losses, dim=0).mean()
        else:
            return torch.tensor(0.0, device=self.dev)

    def forward(self,
                descs1,fb_descs1,kpts1,
                descs2,fb_descs2,kpts2,
                pts,
                alike_kpts1,alike_kpts2,
                DA_normals1,DA_normals2,
                megadepth_batch_size,coco_batch_size
                ):

        # --- 1. 一次性计算所有权重图 ---
        keypoint_weights_s8_list1, smoothness_weights_full_list1 = [], []
        keypoint_weights_s8_list2, smoothness_weights_full_list2 = [], []

        if DA_normals1 is not None and DA_normals2 is not None:
            for b in range(len(DA_normals1)):
                # print(f"Original DA_normals1[{b}] shape: {DA_normals1[b].shape}")
                teacher_n1_chw = DA_normals1[b].to(self.dev)
                # print(f"Permuted teacher_n1_chw shape: {teacher_n1_chw.shape}")
                # 一次调用，得到两种分辨率的权重
                w_kpt1_s8, w_smooth1_full = self._compute_weights_from_teacher_normal(teacher_n1_chw)
                keypoint_weights_s8_list1.append(w_kpt1_s8)
                smoothness_weights_full_list1.append(w_smooth1_full)

                teacher_n2_chw = DA_normals2[b].to(self.dev)
                w_kpt2_s8, w_smooth2_full = self._compute_weights_from_teacher_normal(teacher_n2_chw)
                keypoint_weights_s8_list2.append(w_kpt2_s8)
                smoothness_weights_full_list2.append(w_smooth2_full)

        self.loss_descs,self.acc_coarse,conf_list=self.compute_descriptors_loss(descs1,descs2,pts,keypoint_weights_s8_list1,keypoint_weights_s8_list2)
        # self.loss_descs, self.acc_coarse, conf_list = self.compute_descriptors_loss(descs1, descs2, pts)

        self.loss_kpts,self.acc_kpt=self.compute_keypoints_loss(kpts1,kpts2,alike_kpts1,alike_kpts2,keypoint_weights_s8_list1,keypoint_weights_s8_list2)
        # self.loss_kpts, self.acc_kpt = self.compute_keypoints_loss(kpts1, kpts2, alike_kpts1, alike_kpts2)

        self.loss_normals=self.compute_normals_loss(fb_descs1,fb_descs2,DA_normals1,DA_normals2,smoothness_weights_full_list1,smoothness_weights_full_list2,megadepth_batch_size,coco_batch_size)
        # self.loss_normals = self.compute_normals_loss(fb_descs1, fb_descs2, DA_normals1, DA_normals2)
        return {
            'loss_descs':self.lam_descs*self.loss_descs,
            'acc_coarse':self.acc_coarse,
            'loss_kpts':self.lam_kpts*self.loss_kpts,
            'acc_kpt':self.acc_kpt,
            'loss_normals':self.lam_normals*self.loss_normals,
        }

