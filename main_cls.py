import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from dataset_modelnet import ModelNet40
from dataset_scanobjectnn import ScanObjectNN
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_cls import MetricLoss, Model
from utils.utils import init_folder, seed_everything, cal_loss, IOStream
from utils.utils_args import process_args


def train_one_epoch(args, model, data_loader, opt, criterion, scheduler, epoch=None):
    device = args.device
    train_loss = 0.0
    count = 0.0
    model.train()
    train_pred = []
    train_true = []
    total_time = 0.0
    for combined in tqdm(data_loader):
        start_time = time.time()
        if args.dataset == 'modelnet':
            data, normal, label = combined
            data, normal = data.to(device).float(), normal.to(device).float()
        elif args.dataset == 'scanobject':
            data, label = combined
            data = data.to(device).float()
        else:
            raise TypeError("invalid dataset")
        label = label.to(device).squeeze().long()
        batch_size = data.size()[0]

        opt.zero_grad()

        if args.use_contrast:
            logits, pred_list = model(data[..., :3], normal=None)
            loss = criterion[0](logits, label) + criterion[1](pred_list[-1], pred_list[:-1])
        else:
            logits = model(data[..., :3], normal=None)
            loss = criterion[0](logits, label)

        loss.backward()
        opt.step()

        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        train_true.append(label.cpu().detach().numpy())
        train_pred.append(preds.cpu().detach().numpy())

        end_time = time.time()
        total_time += (end_time - start_time)

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 1e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 1e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 1e-5

    print('train total time is', total_time)
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                             train_loss * 1.0 / count,
                                                                             metrics.accuracy_score(
                                                                                 train_true, train_pred),
                                                                             metrics.balanced_accuracy_score(
                                                                                 train_true, train_pred))
    io.cprint(outstr)


def test_one_epoch(args, model, data_loader, ce_loss, best_test_acc, epoch=None):
    device = args.device
    test_loss = 0.0
    count = 0.0
    test_pred = []
    test_true = []
    total_time = 0.0
    model.eval()
    with torch.no_grad():
        for combined in data_loader:
            start_time = time.time()
            if args.dataset == 'modelnet':
                data, normal, label = combined
                data, normal = data.to(device).float(), normal.to(device).float()
            elif args.dataset == 'scanobject':
                data, label = combined
                data = data.to(device).float()
            label = label.to(device).squeeze().long()
            batch_size = data.size()[0]

            logits = model(data[..., :3], normal=None)
            loss = ce_loss(logits, label)

            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().detach().numpy())
            test_pred.append(preds.cpu().detach().numpy())
            end_time = time.time()
            total_time += (end_time - start_time)

    # args.ratio_test.cprint(str(counter * 1.0 / (count * 512)))
    test_loss = test_loss * 1.0 / count
    print('test total time is', total_time)
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    if args.mode == 'train':
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss,
                                                                              test_acc,
                                                                              avg_per_class_acc)
    else:
        outstr = 'Test loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (test_loss,
                                                                          test_acc,
                                                                          avg_per_class_acc)
    io.cprint(outstr)
    if args.mode == 'train':
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.t7')

        if epoch == args.epochs - 1:
            io.cprint("Best test acc: %.6f\n" % best_test_acc)
            torch.save(model.state_dict(), f'{ckpt_path}/model_best.t7')

        return best_test_acc


def main(args):
    if args.dataset == 'scanobject':
        train_loader = DataLoader(
            ScanObjectNN(train=True, num_points=args.num_points, normalize=True, transforms=True, rotate=args.train_rot_mode),
            num_workers=8, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            ScanObjectNN(train=False, num_points=args.num_points, normalize=True, transforms=False, rotate=args.test_rot_mode),
            num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'modelnet':
        train_loader = DataLoader(
            ModelNet40(train=True, use_normal=False, normalize=False, transforms=True, rotate=args.train_rot_mode),
            num_workers=8, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            ModelNet40(train=False, use_normal=False, normalize=False, transforms=False, rotate=args.test_rot_mode),
            num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise ValueError("unknown dataset")

    args.device = torch.device("cuda" if args.cuda else "cpu")
    device = args.device
    if args.dataset == 'scanobject':
        num_cls = 15
    else:
        num_cls = 40

    # load the model
    model = Model(args, output_channels=num_cls).to(device)

    if args.mode == 'test':
        best_model = f'{ckpt_path}/model_best.t7'
        if os.path.exists(best_model):
            model.load_state_dict(torch.load(best_model))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wt_decay)
    else:
        print("Use Adam")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)

    if args.scheduler == 'cos':
        print("Use CosineAnnealingLR Scheduler")
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100)
    elif args.scheduler == 'step':
        print("Use StepLR Scheduler")
        scheduler = StepLR(opt, step_size=80, gamma=0.1)
    elif args.scheduler == 'other':
        print("Use ReduceLROnPlateau Scheduler")
        scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=args.lr / 100)
    else:
        raise ValueError("no such scheduling method")

    ce_loss = cal_loss
    contrast_loss = MetricLoss()
    best_test_acc = 0

    if args.mode == 'train':
        for epoch in range(args.epochs):
            train_one_epoch(args, model, train_loader, opt, [ce_loss, contrast_loss], scheduler, epoch)
            best_test_acc = test_one_epoch(args, model, test_loader, ce_loss, best_test_acc, epoch)
    else:
        test_one_epoch(args, model, test_loader, ce_loss, best_test_acc)


# def test(args):
#     test_loader = DataLoader(
#         ModelNet40(train=False, use_normal=False, normalize=False, transforms=False, rotate='so3'),
#         num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
#
#     device = torch.device("cuda" if args.cuda else "cpu")
#     num_cls = 40
#     save_embeddings = args.save_embeddings
#     model = Model(args, save_embeddings=save_embeddings, output_channels=num_cls).to(device)
#     model = nn.DataParallel(model)
#
#     args.model_path = 'new_checkpoints/' + 'cross_attention_useContrast_v2_' + 'train/models/model.t7'
#     model.load_state_dict(torch.load(args.model_path))
#     model.eval()
#
#     # for i, (data, normal, label) in enumerate(test_loader):
#     #     data, normal = data.to(device).float(), normal.to(device).float()
#     #     label = label.to(device).squeeze().long()
#     #     logits, xyz = model(data[..., :3], normal=normal)
#     #     preds = logits.max(dim=1)[1]
#     #     if preds == label:
#     #         logits[:, preds].backward()
#     #         # pull the gradients out of the model
#     #         gradients = model.get_activations_gradient()  # (b, C, N)
#     #
#     #         # pool the gradients across the channels
#     #         pooled_gradients = torch.mean(gradients, dim=[0, 2])  # C
#     #
#     #         # get the activations of the last convolutional layer
#     #         activations = model.get_activations(data)[0].detach()
#     #
#     #         # weight the channels by corresponding gradients
#     #         for j in range(1024):
#     #             activations[:, j] *= pooled_gradients[j]
#     #
#     #         # average the channels of the activations
#     #         heatmap = torch.mean(activations, dim=1).squeeze()
#     #         heatmap = F.relu(heatmap)
#     #
#     #         # normalize the heatmap
#     #         heatmap /= torch.max(heatmap)  # (N, 1)
#     #         combined_vis = torch.cat([xyz[0], heatmap[:, None]], dim=-1)  # (N, 4)
#     #         np.savetxt('./attention_map/saved_cam/data_{0}.txt'.format(i), combined_vis.cpu().numpy())
#
#     with torch.no_grad():
#         # if args.add_noise:
#         #     for noise_num in args.noise_point:
#         #         test_pred = []
#         #         test_true = []
#         #         total_time = 0.0
#         #         for data, label in test_loader:
#         #             start_time = time.time()
#         #
#         #             data, label = data.to(device), label.to(device).squeeze()  # [B, N, 3]
#         #             batch_size = data.size()[0]
#         #
#         #             # # create noise points
#         #             # noisy_point = np.random.random((batch_size, noise_num, 3))
#         #             # noisy_point = torch.from_numpy(normalize_data(noisy_point)).to(device)
#         #             # data[:, :noise_num, :3] = noisy_point
#         #             # # data[:, :noise_num, :3] = torch.rand((batch_size, noise_num, 3), device=device)
#         #
#         #             #########################
#         #             data_transpose = data.transpose(2, 1).contiguous()  # [B, 3, N]
#         #             data_ct = data_transpose - data_transpose.mean(dim=-1, keepdim=True)
#         #             data_pca = PCA(args.num_points, data_ct, data_transpose)  # [B, 3, N]
#         #             data_pca = data_pca.transpose(2, 1).contiguous()  # [B, N, 3]
#         #             #########################
#         #
#         #             logits = model(data, data_pca)
#         #             preds = logits.max(dim=1)[1]
#         #             test_true.append(label.cpu().detach().numpy())
#         #             test_pred.append(preds.cpu().detach().numpy())
#         #
#         #             end_time = time.time()
#         #             total_time += (end_time - start_time)
#         #
#         #         print('test total time is', total_time)
#         #         test_true = np.concatenate(test_true)
#         #         test_pred = np.concatenate(test_pred)
#         #         test_acc = metrics.accuracy_score(test_true, test_pred)
#         #         io.cprint('Test with %d noise points:: test acc: %.6f' % (noise_num, test_acc))
#         # if args.add_std:
#         #     for std in args.std_list:
#         #         test_pred = []
#         #         test_true = []
#         #         total_time = 0.0
#         #         for data, label in test_loader:
#         #             start_time = time.time()
#         #
#         #             data, label = data.to(device), label.to(device).squeeze()  # [B, N, 3]
#         #             batch_size = data.size()[0]
#         #
#         #             # create noise
#         #             noise = torch.normal(mean=0, std=std, size=(batch_size, args.num_points, 3)).to(device)
#         #
#         #             np.random.normal(0, std, size=(batch_size, args.num_points, 3))
#         #
#         #             # data_pca += noise
#         #             data += noise
#         #
#         #             #########################
#         #             data_transpose = data.transpose(2, 1).contiguous()  # [B, 3, N]
#         #             data_ct = data_transpose - data_transpose.mean(dim=-1, keepdim=True)
#         #             data_pca = PCA(args.num_points, data_ct, data_transpose)  # [B, 3, N]
#         #             data_pca = data_pca.transpose(2, 1).contiguous()  # [B, N, 3]
#         #             #########################
#         #
#         #             logits = model(data, data_pca)
#         #             preds = logits.max(dim=1)[1]
#         #             test_true.append(label.cpu().detach().numpy())
#         #             test_pred.append(preds.cpu().detach().numpy())
#         #
#         #             end_time = time.time()
#         #             total_time += (end_time - start_time)
#         #
#         #         print('test total time is', total_time)
#         #         test_true = np.concatenate(test_true)
#         #         test_pred = np.concatenate(test_pred)
#         #         test_acc = metrics.accuracy_score(test_true, test_pred)
#         #         io.cprint('Test with %.2f std:: Acc: %.6f' % (std, test_acc))
#         test_pred = []
#         test_true = []
#         total_time = 0.0
#         label_list = []
#         test_embedding = []
#         for i, (data, normal, label) in enumerate(test_loader):
#             start_time = time.time()
#             data, normal = data.to(device).float(), normal.to(device).float()
#             label = label.to(device).squeeze().long()
#             label_list.append(label.view(-1, 1))
#             batch_size = data.size()[0]
#             if save_embeddings:
#                 logits, embedding_list, relation_list = model(data[..., :3], normal=normal)
#                 layer1_embedding = relation_list[0].transpose(-1, -2).contiguous()  # (B, N, 1)
#                 layer2_embedding = relation_list[1].transpose(-1, -2).contiguous()  # (B, N, 1)
#                 layer3_embedding = relation_list[2].transpose(-1, -2).contiguous()  # (B, N, 1)
#                 layer4_embedding = relation_list[3].transpose(-1, -2).contiguous()  # (B, N, 1)
#                 layer_pos = relation_list[4]  # (B, N, 3)
#                 attn = torch.cat([layer_pos, layer1_embedding, layer2_embedding, layer3_embedding, layer4_embedding],
#                                  dim=-1)  # (B, N, 7)
#                 for j in range(batch_size):
#                     np.savetxt('./attention_map/saved_points/data_{0}.txt'.format(i * batch_size + j),
#                                attn[j].cpu().numpy())
#                 # test_embedding.append(embedding_list[-1].cpu().numpy())
#             else:
#                 logits = model(data[..., :3], normal=normal)
#             preds = logits.max(dim=1)[1]
#             test_true.append(label.cpu().detach().numpy())
#             test_pred.append(preds.cpu().detach().numpy())
#             end_time = time.time()
#             total_time += (end_time - start_time)
#
#         print('test total time is', total_time)
#         # if save_embeddings:
#         #     test_embedding = np.concatenate(test_embedding)
#         # test_true = np.concatenate(test_true)
#         # test_pred = np.concatenate(test_pred)
#         # test_acc = metrics.accuracy_score(test_true, test_pred)
#         # io.cprint('test acc: %.6f' % test_acc)
#         #
#         # # visualization
#         # sne_visual(test_embedding, test_pred)


if __name__ == "__main__":
    args = process_args(train=True)
    seed_everything(args)
    exp_path, ckpt_path = init_folder(args)
    io = IOStream(f'{exp_path}/run.log')
    io.cprint(str(args))
    main(args)
