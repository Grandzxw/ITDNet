
import time
import numpy as np
from skimage.metrics import structural_similarity
# from skvideo.measure import niqe

def check_shape_equality(im1, im2):
    """Raise an error if the shape do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return

def _as_floats_riandint(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    # float_type = _supported_float_type([image0.dtype, image1.dtype])
    image0 = np.asarray(image0, dtype=np.float64)
    image1 = np.asarray(image1, dtype=np.float64)
    return image0, image1

def mean_squared_error_riandint(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats_riandint(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def peak_signal_noise_ratio_riandint(image_true, image_test, *, data_range=255):

    check_shape_equality(image_true, image_test)

    image_true, image_test = _as_floats_riandint(image_true, image_test)

    err = mean_squared_error_riandint(image_true, image_test)
    return 10 * np.log10((data_range ** 2) / err)


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape

    # recoverd = recoverd.detach().cpu().numpy()
    # clean = clean.detach().cpu().numpy()
    #
    # recoverd = recoverd.transpose(0, 2, 3, 1)
    # clean = clean.transpose(0, 2, 3, 1)

    # print("recoverd shape", recoverd.shape)
    # print("clean shape", clean.shape)
    recoverd = np.expand_dims(recoverd, axis=0)
    clean = np.expand_dims(clean, axis=0)

    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        # print("recoverd shape_sample", recoverd[i].shape)
        # # 获取两个通道的最大值
        # max_channel_1_recoverd = np.max(recoverd[i][:, :, 0])  # 第一个通道的最大值
        # max_channel_2_recoverd = np.max(recoverd[i][:, :, 1])  # 第二个通道的最大值

        # 打印最大值
        # print("Max value of channel 1:", max_channel_1_recoverd)
        # print("Max value of channel 2:", max_channel_2_recoverd)
        #
        # print("clean shape_sample", clean[i].shape)

        # max_channel_1_clean = np.max(clean[i][:, :, 0])  # 第一个通道的最大值
        # max_channel_2_clean = np.max(clean[i][:, :, 1])  # 第二个通道的最大值
        #
        # # 打印最大值
        # print("Max value of channel 1:", max_channel_1_clean)
        # print("Max value of channel 2:", max_channel_2_clean)

        psnr += peak_signal_noise_ratio_riandint(clean[i], recoverd[i], data_range=255)
        # ssim += structural_similarity(clean[i], recoverd[i], data_range=255, multichannel=True, channels_axis =2)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=255, channel_axis=-1)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0