from cellbin.image.augmentation import f_resize, f_rgb2gray, f_ij_auto_contrast, f_ij_16_to_8
from cellbin.image.threshold import f_th_li

from skimage.exposure import rescale_intensity
import numpy as np


def f_prepocess(img, img_type="ssdna", tar_size=(256, 256)):
    img = np.squeeze(img)
    if img_type == "rna":
        img[img > 0] = 255
        img = np.array(img).astype(np.uint8)
    else:
        img = f_ij_16_to_8(img)
        img = f_rgb2gray(img, True)
    img = f_resize(img, tar_size, "BILINEAR")
    img = f_ij_auto_contrast(img)

    img = np.divide(img, 255.0)
    # img = img.astype('float32')
    # sample_value = img[(0,) * img.ndim]
    # if (img == sample_value).all():
    #     return np.zeros_like(img)
    # img = rescale_intensity(img, out_range=(0.0, 1.0))
    img = np.array(img).astype(np.float32)
    img = np.ascontiguousarray(img)
    return img


def f_postpocess(pred):
    # pred[pred>0.5] = 1.0
    # pred = np.uint8(pred)

    pred = np.uint8(pred * 255)
    pred = f_th_li(pred)
    return pred


def f_preformat(img):
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img


def f_postformat(pred):
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.squeeze(pred)
    return f_postpocess(pred)
