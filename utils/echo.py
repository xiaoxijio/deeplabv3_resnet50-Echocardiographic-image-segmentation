import os
import collections
import pandas

import numpy as np
import skimage.draw
import torchvision

from utils.utils import loadvideo


class Echo(torchvision.datasets.VisionDataset):
    def __init__(self, root=None, split="train", target_type="EF", mean=0., std=1., length=16, period=2,
                 max_length=250, clips=1, pad=None, noise=None, target_transform=None, external_test_location=None):
        """
        超声心动数据集
        :param root: 数据集的根目录
        :param split: One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        :param target_type: 目标类型
        ``Filename'' (string): 视频的文件名
        ``EF'' (float): 射血分数
        ``EDV'' (float): 舒张末期容积
        ``ESV'' (float): 收缩末期容积
        ``LargeIndex'' (int): 视频中对应舒张（大）帧的索引
        ``SmallIndex'' (int): 视频中对应收缩（小）帧的索引
        ``LargeFrame'' (np. array shape=(3, h, w)): 归一化的舒张（大）帧图像
        ``SmallFrame'' (np. array shape=(3, h, w)): 归一化的收缩（小）帧图像
        ``LargeTrace'' (np. array shape=(h, w)): 左心室舒张（大）帧的分割结果 (值为 0 表示该像素在左心室外部, 1 在左心室内部)
        ``SmallTrace'' (np.  array shape=(h, w)): 左心室收缩（小）帧的分割结果(值为 0 表示该像素在左心室外部, 1 在左心室内部)

        :param mean: 如果是标量,表示对所有通道使用的均值. 如果是 NP 数组，表示对每个通道分别使用的均值
        :param std: 如果是标量，表示对所有通道使用的标准差. 如果是 NP 数组，表示对每个通道分别使用的标准差
        :param length: 从视频中剪辑的帧数。如果为 None，返回最长可能的剪辑
        :param period: 每隔 period 帧取一帧
        :param max_length: 从视频中剪辑的最大帧数. 如果为 None，不会对任何视频进行缩短
        :param clips: 采样的剪辑数量. 主要用于测试时的增强，通过随机剪辑实现
        :param pad: 填充
        :param noise: 模拟噪声的黑化像素比例
        :param target_transform: 数据增强函数
        :param external_test_location: 用于外部测试的视频路径
        """
        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":  # eg: 分离出train数据
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # filename.avi
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)  # 创建特殊的字典，它允许在访问不存在的键时自动创建一个默认值
            self.trace = collections.defaultdict(_defaultdict_of_lists)  # 嵌套的字典结构

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:  # 去除重复的frame
                        self.frames[filename].append(frame)  # {filename：frame}
                    self.trace[filename][frame].append((x1, y1, x2, y2))  # {filename：{frame:[(x1, y1, x2, y2)]}}
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # 少数视频缺失痕迹；删除这些视频
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = loadvideo(video).astype(np.float32)

        # 添加模拟噪声（黑掉随机像素） 0表示黑色（视频还没有归一化）
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # 归一化
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # 如果视频太短，就用0填充帧 0代表平均颜色（深灰色），因为这是在标准化之后
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # 随机提一帧作为开始
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(int(self.frames[key][0]))
            elif t == "LargeFrame":  # frames[key]里会存两帧 舒张和收缩的多边形坐标
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:  # video的mask，舒张或收缩区域为1 其余地方为0
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))  # 将多边形的坐标 (x1, y1, x2, y2) 连接并翻转
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                # 根据提供的 x 和 y 坐标绘制一个多边形 r：多边形边界的行索引 c：多边形边界的列索引
                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1  # 多边形区域被标记为 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # 添加零填充（视频的平均颜色）
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)  # 裁剪
            video = temp[:, :, i:(i + h), j:(j + w)]

        return video, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)




