# Author LGF
# GreatTime 2020/5/19
# Description: simple introduction of the code
# 参考https://blog.csdn.net/qq_36219202/article/details/78339459
import os
import cv2
from PIL import Image, ImageEnhance
import skimage
import random
import numpy as np


# 随机镜像
def random_mirror(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    filp_img = np.asarray(filp_img, dtype="float32")
    return filp_img


# 随机平移
def random_move(root_path, img_name, off):
    img = Image.open(os.path.join(root_path, img_name))
    offset = img.offset(off, 0)
    offset = np.asarray(offset, dtype="float32")
    return offset


# # 随机转换
# def random_transform( image, rotation_range, zoom_range, shift_range, random_flip ):
#     h,w = image.shape[0:2]
#     rotation = numpy.random.uniform( -rotation_range, rotation_range )
#     scale = numpy.random.uniform( 1 - zoom_range, 1 + zoom_range )
#     tx = numpy.random.uniform( -shift_range, shift_range ) * w
#     ty = numpy.random.uniform( -shift_range, shift_range ) * h
#     mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
#     mat[:,2] += (tx,ty)
#     result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
#     if numpy.random.random() < random_flip:
#         result = result[:,::-1]
#     return result


# # 随机变形
# def random_warp( image ):
#     assert image.shape == (256,256,3)
#     range_ = numpy.linspace( 128-80, 128+80, 5 )
#     mapx = numpy.broadcast_to( range_, (5,5) )
#     mapy = mapx.T
#
#     mapx = mapx + numpy.random.normal( size=(5,5), scale=5 )
#     mapy = mapy + numpy.random.normal( size=(5,5), scale=5 )
#
#     interp_mapx = cv2.resize( mapx, (80,80) )[8:72,8:72].astype('float32')
#     interp_mapy = cv2.resize( mapy, (80,80) )[8:72,8:72].astype('float32')
#
#     warped_image = cv2.remap( image, interp_mapx, interp_mapy, cv2.INTER_LINEAR )
#
#     src_points = numpy.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
#     dst_points = numpy.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
#     mat = umeyama( src_points, dst_points, True )[0:2]
#
#     target_image = cv2.warpAffine( image, mat, (64,64) )
#
#     return warped_image, target_image


# 随机旋转
def random_rotate(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(180)
    rotation_img = np.asarray(rotation_img, dtype="float32")
    return rotation_img


# # 随机裁剪
# def random_clip(root_path, floder, imagename):
#     # 可以使用crop_img = tf.random_crop(img,[280,280,3])
#     img = cv2.imread(root_path + floder + imagename)
#     count = 1  # 随机裁剪的数量
#     while 1:
#         y = random.randint(1, 8)
#         x = random.randint(1, 8)
#         cropImg = img[(y):(y + 120), (x):(x + 120)]
#         image_save_name = root_path + floder + 'clip' + str(count) + imagename
#         # BGR2RGB
#         # cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
#         cropImg = cv2.resize(cropImg, (48, 48))
#         cv2.imwrite(image_save_name, cropImg)
#         count += 1
#         print(count)
#         if count > 3:
#             break


# 随机噪声
def random_noise(root_path, img_name):
    image = Image.open(os.path.join(root_path, img_name))
    im = np.array(image)

    means = 0
    sigma = 10

    r = im[:, :, 0].flatten()
    g = im[:, :, 1].flatten()
    b = im[:, :, 2].flatten()

    # 计算新的像素值
    for i in range(im.shape[0] * im.shape[1]):
        pr = int(r[i]) + random.gauss(means, sigma)
        pg = int(g[i]) + random.gauss(means, sigma)
        pb = int(b[i]) + random.gauss(means, sigma)

        if (pr < 0):
            pr = 0
        if (pr > 255):
            pr = 255
        if (pg < 0):
            pg = 0
        if (pg > 255):
            pg = 255
        if (pb < 0):
            pb = 0
        if (pb > 255):
            pb = 255
        r[i] = pr
        g[i] = pg
        b[i] = pb
    im[:, :, 0] = r.reshape([im.shape[0], im.shape[1]])
    im[:, :, 1] = g.reshape([im.shape[0], im.shape[1]])
    im[:, :, 2] = b.reshape([im.shape[0], im.shape[1]])
    gaussian_image = Image.fromarray(np.uint8(im))
    return gaussian_image


# 随机调整对比度
def random_adj(root_path, img_name):
    image = skimage.io.imread(os.path.join(root_path, img_name))
    gam = skimage.exposure.adjust_gamma(image, 0.5)
    log = skimage.exposure.adjust_log(image)
    gam = np.asarray(gam, dtype="float32")
    log = np.asarray(log, dtype="float32")
    return gam, log


# 运行
def main():
    root_dir = "flower_photos/"
    floder = "women/"

    images = os.listdir(root_dir + floder)

    for imagename in images:
        mirror_img = random_mirror(root_dir + floder, imagename)
        image_save_name = root_dir + floder + "mirror" + imagename
        mirror_img = cv2.cvtColor(mirror_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_save_name, mirror_img)

        # random_clip(root_dir, floder, imagename)

        noise_img = random_noise(root_dir + floder, imagename)
        noise_img = np.asarray(noise_img, dtype="float32")
        image_save_name = root_dir + floder + "noise" + imagename
        noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_save_name, noise_img)

    print("image preprocessing")


if __name__ == '__main__':
    main()
