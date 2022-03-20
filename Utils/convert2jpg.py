import rawpy
import imageio
import glob
from PIL import Image
import os
import numpy as np

def raw2jpg(raw_img_file, dst="", _suffix=".NEF"):
    """
    :param raw_img_file : 转换前原图
    :param dst : 转换后jpg存储目录
    :param _suffix : 原图文件后缀
    :return :
    """
    with rawpy.imread(raw_img_file) as raw:
        im = raw.postprocess(
            use_camera_wb=True, # 是否使用拍摄时的白平衡值
            use_auto_wb=False,
            # 修改后光线会下降，所以需要手动提亮，线性比例的曝光偏移。可用范围从0.25（变暗2级）到8.0（变浅3级）。
            exp_shift=3 
        )
        # 因为glob函数返回的是一个相对路径，所以不需要使用os.path
        imageio.imsave(dst + raw_img_file.strip(_suffix) + ".jpg", im)  


def tif2jpg(tif_img_file, dst="depthJPG/", _suffix="tif"):
    def preprocess_sfm_depth_map(depths, min_depth=0.1, max_depth=1.0):
        z_min = np.min(depths) + (min_depth * (np.max(depths) - np.min(depths)))
        z_max = np.min(depths) + (max_depth * (np.max(depths) - np.min(depths)))
        if max_depth != 0:
            depths[depths == 0] = z_max
        depths[depths < z_min] = 0
        return depths
    with Image.open(tif_img_file) as img:
        img = preprocess_sfm_depth_map(np.array(img))
        img = Image.fromarray(img)
        img.save(os.path.join(dst, tif_img_file.strip(_suffix) + ".jpg"))

if __name__ == '__main__':
    src_suffix = ".tif"
    dst_suffix = ".jpg"
    raw_files = glob.glob(f"*{src_suffix}")

    print("正在转换中，请耐心等待....")
    for num, raw_file in enumerate(raw_files):
        if num % 5 == 0: print(f"已转换{num}张照片...")
        tif2jpg(raw_file, _suffix=src_suffix)

    print("转换完成！")
