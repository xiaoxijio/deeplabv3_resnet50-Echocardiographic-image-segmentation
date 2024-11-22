import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import Subset, DataLoader

from utils.echo import Echo
from utils.utils import get_mean_and_std, bootstrap, dice_similarity_coefficient, latexify, savevideo


@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default='data/EchoNet-Dynamic')
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if
           name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
              default="deeplabv3_resnet50")
@click.option("--pretrained/--random", default=False)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=True)
@click.option("--save_video/--skip_video", default=True)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-5)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=0)  # windows不要动
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(data_dir=None, output=None, model_name="deeplabv3_resnet50", pretrained=False, weights=None, run_test=False,
        save_video=False, num_epochs=50, lr=1e-5, weight_decay=1e-5, lr_step_period=None, num_train_patients=None,
        num_workers=0, batch_size=20, device=None, seed=0):
    """

    :param data_dir: 包含数据集的目录
    :param output: 存放输出的目录名
    :param model_name: 分割模型的名称。“deeplabv3_resnet50”，“deeplabv3_resnet101”
    :param pretrained: 是否为模型使用预训练的权重
    :param weights: 包含初始化模型权重的检查点的路径
    :param run_test: 是否运行测试
    :param save_video: 是否保存带有分段的视频
    :param num_epochs: 训练期间的epoch数量
    :param lr: SGD的学习率
    :param weight_decay: SGD的权重衰减
    :param lr_step_period: 学习率衰减的周期 math.Inf（永不衰减学习率）
    :param num_train_patients: 消融
    :param num_workers: windows不要动
    :param batch_size: 每批要加载多少个样本
    :param device: 运行的设备
    :param seed: 随机数生成器的种子

    :return:
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation",
                              "{}_{}".format(model_name, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=False)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[
        -1].kernel_size)  # 因为是直接调用模型，所以需要改一下输出层的数量
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # 因为这个数据集的特殊性，没法应用常规的均值和方差，所以需要自己算
    mean, std = get_mean_and_std(Echo(root=data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks, "mean": mean, "std": std}

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = Echo(root=data_dir, split="train", **kwargs)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = Subset(dataset["train"], indices)
    dataset["val"] = Echo(root=data_dir, split="val", **kwargs)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):  # 单机单卡流下泪水
                    torch.cuda.reset_peak_memory_stats(i)  # 清除每个 GPU 的内存使用峰值数据

                ds = dataset[phase]
                dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                        pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader,
                                                                                     phase == "train",
                                                                                     optim, device)
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (
                        large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch, phase, loss, overall_dice, large_dice,
                                                                    small_dice, time.time() - start_time,
                                                                    large_inter.size,
                                                                    sum(torch.cuda.max_memory_allocated() for i in
                                                                        range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_reserved() for i in
                                                                        range(torch.cuda.device_count())),
                                                                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': bestLoss,
                    'loss': loss,
                    'opt_dict': optim.state_dict(),
                    'scheduler_dict': scheduler.state_dict()}
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            # Run on validation and test
            for split in ["val", "test"]:
                dataset = Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                                         pin_memory=(device.type == "cuda"))
                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader, False, None,
                                                                                     device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)

                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                s1, s2, s3 = bootstrap(np.concatenate((large_inter, small_inter)),
                                       np.concatenate((large_union, small_union)), dice_similarity_coefficient)
                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, s1, s2, s3))
                s1, s2, s3 = bootstrap(large_inter, large_union, dice_similarity_coefficient)
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, s1, s2, s3))
                s1, s2, s3 = bootstrap(small_inter, small_union, dice_similarity_coefficient)
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, s1, s2, s3))
                f.flush()

    # Saving videos with segmentations
    dataset = Echo(root=data_dir, split="test", target_type=["Filename", "LargeIndex", "SmallIndex"],
                   # Need filename for saving, and human-selected frames to annotate
                   mean=mean, std=std,  # Normalization
                   length=None, max_length=None, period=1  # Take all frames
                   )
    dataloader = DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False,
                            pin_memory=False, collate_fn=_video_collate_fn)

    # Save videos with segmentation
    if save_video and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        # Only run if missing videos

        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate(
                        [model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in
                         range(0, x.shape[0], batch_size)])

                    start = 0
                    x = x.numpy()
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            g.write(
                                "{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0,
                                                             1 if frame == small_index[i] else 0,
                                                             1 if frame in systole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for (f, s) in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf

                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape(
                                    (1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape(
                                    (1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))),
                                                     4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video
                        start += offset


def run_epoch(model, dataloader, train, optim, device):
    """
    运行一个 epoch 的 训练 / 评估
    :param model: 要 训练 / 评估 的模型
    :param dataloader: 数据集
    :param train: 是否训练模型
    :param optim: 优化器
    :param device: 要运行的设备
    :return:
    """
    total, n, pos, neg, pos_pix, neg_pix = 0., 0, 0, 0, 0, 0
    model.train(train)

    large_inter, large_union, small_inter, small_union = 0, 0, 0, 0
    large_inter_list, large_union_list, small_inter_list, small_union_list = [], [], [], []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                # 计算图像中像素的正负类别 (值为 0 表示该像素在左心室外部, 1 在左心室内部)
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                # 计算每一帧中每个像素在真实标签中的正负分类
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # 对舒张期帧进行预测并计算损失
                large_frame, large_trace = large_frame.to(device), large_trace.to(device)
                y_large = model(large_frame)["out"]
                loss_large = binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")

                # 计算心脏舒张期帧的交并比 (IoU)
                # large_inter 是交集，表示模型预测为 1 且真实标签为 1 的像素数
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0.,
                                              large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                # large_union 是并集，表示模型预测为 1 或真实标签为 1 的像素数
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0.,
                                             large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0.,
                                                       large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0.,
                                                      large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # 对收缩期帧进行预测并计算损失
                small_frame, small_trace = small_frame.to(device), small_trace.to(device)
                y_small = model(small_frame)["out"]
                loss_small = binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")

                # 计算收缩期帧的交并比 (IoU)
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0.,
                                              small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0.,
                                             small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0.,
                                                       small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0.,
                                                      small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                n += large_trace.size(0)
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on process bar
                s1 = total / n / 112 / 112  # 每像素的平均损失
                s2 = loss.item() / large_trace.size(0) / 112 / 112  # 每帧的平均损失
                s3 = -p * math.log(p) - (1 - p) * math.log(1 - p)  # 信息熵
                s4 = (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean()  # 每像素的熵
                s5 = 2 * large_inter / (large_union + large_inter)  # 计算舒张期和收缩期的IoU
                s6 = 2 * small_inter / (small_union + small_inter)
                s = (
                    "平均损失:{:.4f} ({:.4f}) / 信息熵:{:.4f} ({:.4f}), 舒张期IoU:{:.4f}, 收缩期IoU:{:.4f}".format(
                        s1, s2, s3, s4, s5, s6))
                pbar.set_description(s)
                pbar.update()

    loss = total / n / 112 / 112
    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return loss, large_inter_list, large_union_list, small_inter_list, small_union_list


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    # eg:
    # video1.shape = (3, 100, 112, 112)  # 100 帧
    # video2.shape = (3, 120, 112, 112)  # 120 帧
    # i = [100, 120]
    i = list(map(lambda t: t.shape[1], video))  # 获取每个视频的帧数

    # eg:
    # video1.shape = (3, 100, 112, 112)
    # video2.shape = (3, 120, 112, 112)
    # 拼接后: (3, 220, 112, 112)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))  # 拼接视频数据

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i


run()
