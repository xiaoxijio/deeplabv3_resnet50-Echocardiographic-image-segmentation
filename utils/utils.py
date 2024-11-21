import os
import typing

import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, Subset, DataLoader


def get_mean_and_std(dataset, samples=128, batch_size=8, num_workers=0):
    """
    计算数据集的均值和方差
    :param dataset: 数据集 'dataset[i][0]表示数据集中的第 i个视频，维度为(通道，帧，高度，宽度)
    :param samples: 样本数量
    :param batch_size: 每批要加载多少个样本
    :param num_workers: windows设为 0
    :return: 均值和标准差的元组
    """

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = Subset(dataset, indices)  # 根据indices从数据集里提取数据
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n, s1, s2 = 0, 0., 0.  # 累积图像中所有像素的数量  累积所有像素值的和（按通道） 累积所有像素值的平方和（按通道）
    for (x, *_) in tqdm.tqdm(dataloader):  # 将数据展平为每个通道的像素向量
        x = x.transpose(0, 1).contiguous().view(3, -1)  # (b, c, h, w) -> (c, b, h * w) -> (3, num_pixels)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()  # 计算每个通道的像素总和
        s2 += torch.sum(x ** 2, dim=1).numpy()  # 计算每个通道像素值的平方和
    mean = s1 / n  # 每个通道的均值
    std = np.sqrt(s2 / n - mean ** 2)  # 每个通道的标准差

    return mean.astype(np.float32), std.astype(np.float32)


def loadvideo(filename: str) -> np.ndarray:
    """
    从文件加载视频
    :param filename: 视频文件名
    :return: (channels=3, frames, height, width)
    FileNotFoundError: 找不到 "filename"
    ValueError: 读取视频时发生错误
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取每一帧的宽度
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取每一帧的高度

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):  # 循环读取每一帧
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        v[count, :, :] = frame

    v = v.transpose((3, 0, 1, 2))  # (f, h, w, c) -> (c, f, h, w)

    return v


def bootstrap(a, b, func, samples=10000):
    """
    计算 func(a, b) 的自举置信区间
    :param a:
    :param b:
    :param func: 计算置信区间的函数 dataset[i][0}是数据集中的第 i个视频 (channels=3, frames, height, width)
    :param samples: 要计算的样本数量
    :return:
    """
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for _ in range(samples):
        ind = np.random.choice(len(a), len(a))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps)

    return func(a, b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


def dice_similarity_coefficient(inter, union):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """
    return 2 * sum(inter) / (sum(union) + sum(inter))


def latexify():
    """Sets matplotlib params to appear more like LaTeX.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html
    """
    params = {'backend': 'pdf',
              'axes.titlesize': 8,
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)


def savevideo(filename: str, array: np.ndarray, fps: typing.Union[float, int] = 1):
    """Saves a video to a file.

    Args:
        filename (str): filename of video
        array (np.ndarray): video of uint8's with shape (channels=3, frames, height, width)
        fps (float or int): frames per second

    Returns:
        None
    """

    c, _, height, width = array.shape

    if c != 3:
        raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(
            ", ".join(map(str, array.shape))))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in array.transpose((1, 2, 3, 0)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
