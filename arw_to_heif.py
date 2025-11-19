#!/usr/bin/env python3

"""
ARW 转 HEIF 工具

功能：
- 读取 Sony ARW 原始文件
- 转换为线性 sRGB 色彩空间
- 支持按长边限制缩小图像
- 智能曝光分析与调整
- 自然饱和度增强（默认开启）
- 输出 PQ HDR HEIF 格式（10-bit）

处理流程：
1. 读取 ARW（应用相机白平衡）
2. 转换为线性 sRGB 空间
3. 智能曝光分析与调整（可选）
4. 自然饱和度增强（默认 1.3x）
5. 缩小图像（可选）
6. 色域转换 (sRGB→BT.2020)
7. PQ 编码
8. 输出 HEIF

HDR 输出说明（符合 HDR10 标准）：
- 色域：BT.2020 (ITU-R BT.2020)
- 传输函数：PQ (SMPTE ST 2084)
- 支持自定义峰值亮度
- HEIF 容器，高压缩率

针对高动态范围相机（如 Sony A7R4）的优化参数：
- vibrance: 1.3（自然饱和度增强，默认启用）
- diffuse-nits: 700（普通白色亮度）
- peak-nits: 1800（高光峰值亮度）
- hdr-threshold: 0.65（高光阈值）
- 支持 --auto-ev 自动曝光调整

使用示例：
  # 基础使用（推荐用于 A7R4，默认启用 1.3x 自然饱和度）
  python arw_to_heif.py -i photo.ARW --auto-ev

  # 自定义饱和度强度
  python arw_to_heif.py -i photo.ARW --vibrance 1.5

  # 禁用饱和度增强
  python arw_to_heif.py -i photo.ARW --vibrance 1.0

  # 手动曝光补偿 + 自定义饱和度
  python arw_to_heif.py -i photo.ARW --ev 0.5 --vibrance 1.4

  # 自定义 HDR 参数
  python arw_to_heif.py -i photo.ARW --diffuse-nits 700 --peak-nits 1800
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import rawpy
from pillow_heif import register_heif_opener

register_heif_opener()
DEBUG = bool(os.getenv('DEBUG'))


def stat_image(rgb_float):
    print("\n线性光数据统计:")
    print(f"最小值: {rgb_float.min():.6f}")
    print(f"最大值: {rgb_float.max():.6f}")
    print(f"\n平均值: {rgb_float.mean():.6f}")
    print(f"中位数: {np.median(rgb_float):.6f}")
    print(f"标准差: {rgb_float.std():.6f}")

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print("\n百分位数分布:")
    for p in percentiles:
        value = np.percentile(rgb_float, p)
        print(f"  {p:2d}%: {value:.6f}")

    over_one_ratio = (rgb_float > 1.0).sum() / rgb_float.size * 100
    print(f"\n超过 1.0 的像素占比: {over_one_ratio:.4f}%")

    if over_one_ratio > 0:
        print(f"最大超出值: {rgb_float[rgb_float > 1.0].max():.6f}")

    per_channel_stats = []
    for i, channel_name in enumerate(['R', 'G', 'B']):
        channel_data = rgb_float[:, :, i]
        stats = {
            'name': channel_name,
            'min': channel_data.min(),
            'max': channel_data.max(),
            'mean': channel_data.mean(),
            'median': np.median(channel_data),
        }
        per_channel_stats.append(stats)

    print('\n' + '=' * 60)
    print("各通道统计:")
    for stats in per_channel_stats:
        print(
            f"  {stats['name']} 通道 - 最小: {stats['min']:.6f}, 最大: {stats['max']:.6f}, "
            f"平均: {stats['mean']:.6f}, 中位数: {stats['median']:.6f}"
        )


def find_arw_files(directory):
    """
    遍历目录，查找所有 ARW 文件（仅一级子目录）

    Args:
        directory: 目录路径

    Returns:
        ARW 文件路径列表
    """
    arw_files = []
    dir_path = Path(directory)

    if not dir_path.is_dir():
        arw_files.append(str(dir_path))
        return arw_files

    for file_path in dir_path.glob('*.arw'):
        if file_path.is_file():
            arw_files.append(str(file_path))

    for file_path in dir_path.glob('*.ARW'):
        if file_path.is_file() and str(file_path) not in arw_files:
            arw_files.append(str(file_path))

    arw_files.sort()
    return arw_files


def process_arw(arw_path, ev=0.0, auto_bright=False, use_camera_wb=True):
    """
    处理 ARW 文件，读取为线性 sRGB

    Args:
        arw_path: ARW 文件路径
        exposure_compensation: 曝光补偿（EV值），正值增加亮度，负值降低亮度
        auto_bright: 是否启用自动亮度调整
        use_camera_wb: 是否使用相机白平衡

    Returns:
        处理后的线性 RGB 图像
    """
    print(f"\n读取文件: {arw_path}")

    with rawpy.imread(arw_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=use_camera_wb,
            use_auto_wb=not use_camera_wb,
            no_auto_bright=not auto_bright,
            output_bps=16,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.sRGB,
        )

    rgb_float = rgb.astype(np.float32) / 65535.0

    if ev != 0.0:
        exposure_factor = 2.0**ev
        rgb_float = rgb_float * exposure_factor
        print(f"应用曝光补偿: {ev:+.1f} EV (系数: {exposure_factor:.2f})")

    if DEBUG:
        stat_image(rgb_float)

    return rgb_float


def analyze_and_adjust_exposure(rgb_float, target_middle_gray=0.18, auto_adjust=False):
    """
    分析图像并自动调整曝光

    Args:
        rgb_float: 线性 RGB 图像
        target_middle_gray: 目标中灰亮度（线性空间），默认 0.18
        auto_adjust: 是否自动调整

    Returns:
        调整后的图像，调整系数
    """
    luminance = 0.2126 * rgb_float[:, :, 0] + 0.7152 * rgb_float[:, :, 1] + 0.0722 * rgb_float[:, :, 2]

    luminance_clipped = np.clip(luminance, 0.0001, 1.0)
    median_lum = np.median(luminance_clipped)
    mean_lum = np.mean(luminance_clipped)
    p95_lum = np.percentile(luminance_clipped, 95)

    print(f"\n图像亮度分析:")
    print(f"  中位数亮度: {median_lum:.4f}")
    print(f"  平均亮度: {mean_lum:.4f}")
    print(f"  95% 分位数: {p95_lum:.4f}")

    if auto_adjust:
        if median_lum < 0.05:
            adjust_factor = target_middle_gray / max(median_lum, 0.01)
            adjust_factor = min(adjust_factor, 4.0)
            ev_adjustment = np.log2(adjust_factor)

            print(f"  检测到欠曝，建议曝光补偿: +{ev_adjustment:.2f} EV (系数: {adjust_factor:.2f})")

            rgb_adjusted = rgb_float * adjust_factor
            return rgb_adjusted, adjust_factor
        elif median_lum > 0.35:
            adjust_factor = target_middle_gray / median_lum
            ev_adjustment = np.log2(adjust_factor)

            print(f"  检测到过曝，建议曝光补偿: {ev_adjustment:.2f} EV (系数: {adjust_factor:.2f})")

            rgb_adjusted = rgb_float * adjust_factor
            return rgb_adjusted, adjust_factor
        else:
            print(f"  曝光正常，无需调整")
            return rgb_float, 1.0
    else:
        if median_lum < 0.05:
            ev_suggestion = np.log2(target_middle_gray / max(median_lum, 0.01))
            ev_suggestion = min(ev_suggestion, 2.0)
            print(f"  建议增加曝光补偿: +{ev_suggestion:.2f} EV")
        elif median_lum > 0.35:
            ev_suggestion = np.log2(target_middle_gray / median_lum)
            print(f"  建议降低曝光补偿: {ev_suggestion:.2f} EV")

        return rgb_float, 1.0


def enhance_vibrance(rgb_float, strength=1.3):
    """
    自然饱和度增强（Vibrance）

    自然饱和度相比普通饱和度的优势：
    1. 对低饱和度颜色增强更多
    2. 对高饱和度颜色增强较少，避免过饱和
    3. 对肤色等中性色保护较好

    Args:
        rgb_float: 线性 RGB 图像
        strength: 饱和度增强强度，1.0 表示不变，1.3-1.5 为推荐范围

    Returns:
        增强后的图像
    """
    if strength == 1.0:
        return rgb_float

    print(f"\n应用自然饱和度增强（强度: {strength:.2f}）...")

    rgb_gamma = linear_to_srgb_gamma(rgb_float)
    hsv_image = cv2.cvtColor(rgb_gamma, cv2.COLOR_RGB2HSV)

    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    saturation_mask = 1.0 - s
    saturation_boost = 1.0 + (strength - 1.0) * saturation_mask

    s_enhanced = np.clip(s * saturation_boost, 0, 1.0)

    hsv_image[:, :, 1] = s_enhanced

    rgb_gamma_enhanced = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    rgb_linear_enhanced = srgb_gamma_to_linear(rgb_gamma_enhanced)

    return rgb_linear_enhanced


def linear_to_srgb_gamma(linear_rgb):
    """将线性 RGB 转换为 sRGB gamma 编码"""
    srgb = np.where(
        linear_rgb <= 0.0031308, 12.92 * linear_rgb, 1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055
    ).astype(np.float32)
    return np.clip(srgb, 0, 1)


def srgb_gamma_to_linear(srgb):
    """将 sRGB gamma 编码转换为线性 RGB"""
    linear = np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4)).astype(np.float32)
    return linear


def resize_image(image_float, max_long_edge):
    """
    按长边限制缩小图像

    Args:
        image_float: 线性 RGB 图像
        max_long_edge: 长边最大像素数

    Returns:
        缩小后的图像
    """
    height, width = image_float.shape[:2]
    current_long_edge = max(height, width)

    if current_long_edge <= max_long_edge:
        print(f"图像尺寸 {width}x{height}，无需缩小")
        return image_float

    scale = max_long_edge / current_long_edge
    new_width = int(width * scale)
    new_height = int(height * scale)
    print(f"缩小图像: {width}x{height} → {new_width}x{new_height} (缩放比例: {scale:.3f})")
    resized = cv2.resize(image_float, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def srgb_to_bt2020(linear_srgb):
    """
    将线性 sRGB 转换到线性 BT.2020 色域

    使用 sRGB D65 到 BT.2020 D65 的色域转换矩阵
    """
    print("转换色域: sRGB → BT.2020...")

    matrix_srgb_to_bt2020 = np.array(
        [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]], dtype=np.float32
    )

    original_shape = linear_srgb.shape
    rgb_flat = linear_srgb.reshape(-1, 3)

    bt2020_flat = rgb_flat @ matrix_srgb_to_bt2020.T

    bt2020 = bt2020_flat.reshape(original_shape)

    return bt2020


def linear_to_pq(bt2020_image, *, max_nits=1000.0, hdr_threshold=0.0, diffuse_nits=500.0, peak_nits=1500.0):
    """
    将线性 RGB 转换为 PQ (Perceptual Quantizer) HDR

    PQ 是 SMPTE ST 2084 / ITU-R BT.2100 标准定义的 EOTF
    用于 HDR10 内容

    Args:
        bt2020_image: 线性光值（BT.2020 色域），范围 [0, inf)
        max_nits: 内容的最大亮度（尼特），默认 1000 nits（仅在 hdr_threshold=0 时使用）

        hdr_threshold: HDR 高光阈值，大于 0 时启用 fake HDR 效果
        diffuse_nits: 普通白色对应的亮度（nits），仅在 fake HDR 模式下使用
        peak_nits: 高光峰值亮度（nits），仅在 fake HDR 模式下使用

    Returns:
        PQ 编码值，范围 [0, 1]
    """
    pq_max_nits = 10000.0
    if hdr_threshold > 0:
        print(f"应用 PQ HDR 编码（Fake HDR 模式: 漫反射={diffuse_nits} nits, 峰值={peak_nits} nits）...")
        fake_nits = apply_fake_hdr_effect(
            bt2020_image, diffuse_nits=diffuse_nits, peak_nits=peak_nits, threshold=hdr_threshold
        )
        normalized = fake_nits / pq_max_nits
    else:
        print(f"应用 PQ HDR 编码（标准模式: 峰值亮度={max_nits} nits）...")
        normalized = np.clip(bt2020_image * (max_nits / pq_max_nits), 0, 1)

    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0

    normalized_m1 = np.power(normalized, m1)
    pq = np.power((c1 + c2 * normalized_m1) / (1.0 + c3 * normalized_m1), m2)

    return np.clip(pq, 0, 1)


def apply_fake_hdr_effect(linear_img, *, diffuse_nits=203, peak_nits=1000, threshold=0.7):
    """
    逆色调映射：制造 HDR 效果

    Args:
        linear_img: 线性 RGB 图像 (0.0 - 1.0)
        diffuse_nits: 普通白色对应的亮度 (建议 203 - 300, 保证发热后也能看清内容)
        peak_nits: 你希望高光达到的最高亮度 (建议 1200 - 1500, 适配 90% 的旗舰机真实能力，避免 Tone Mapping 导致的整体压暗)
        threshold: 超过多少亮度的像素开始被视为高光 (0.0 - 1.0, 建议 0.6 - 0.7, 只提亮真正的发光点，减少全屏高亮导致的发热加速)
    """
    nits_img = linear_img * diffuse_nits

    luminance = 0.2627 * linear_img[:, :, 0] + 0.6780 * linear_img[:, :, 1] + 0.0593 * linear_img[:, :, 2]

    if DEBUG:
        print('最高亮度:', luminance.max())
        print()

    mask = np.clip((luminance - threshold) / (1.0 - threshold), 0, 1)
    mask = np.power(mask, 2)

    extra_boost_factor = peak_nits / diffuse_nits

    final_nits = nits_img + (nits_img * mask[:, :, np.newaxis] * (extra_boost_factor - 1))
    return final_nits


def save_heif_10bit(image_float, output_path):
    """
    保存 HDR10 HEIF 图像

    Args:
        image_float: PQ 编码的图像数据
        output_path: 输出路径
    """
    print(f"保存 HEIF 文件到: {output_path}")
    print(f"输入图像形状: {image_float.shape}, 数据类型: {image_float.dtype}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pq_int16 = (image_float * 65535.0).clip(0, 65535).astype(np.uint16)

    height, width, channels = pq_int16.shape

    heif_file = pillow_heif.from_bytes(
        mode="RGB;16",
        size=(width, height),
        data=pq_int16.tobytes(),
    )

    nclx_profile = {
        "color_primaries": 9,
        "transfer_characteristics": 16,
        "matrix_coefficients": 9,
        "full_range_flag": True,
    }
    encoder_params = {"chroma": "422"}

    heif_file.save(output_path, quality=-1, bit_depth=10, nclx_profile=nclx_profile, enc_params=encoder_params)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ HEIF 文件保存完成: {file_size_mb:.2f} MB")


def process_single_arw(
    arw_path,
    hdr_output_path=None,
    max_nits=203.0,
    hdr_threshold=0.6,
    max_long_edge=None,
    ev=0.0,
    diffuse_nits=500.0,
    peak_nits=1500.0,
    auto_ev=False,
    auto_bright=False,
    use_camera_wb=True,
    vibrance=1.3,
):
    """
    处理单个 ARW 文件

    Args:
        arw_path: ARW 文件路径
        hdr_output_path: HDR HEIF 输出路径（可选）
        max_nits: HDR 峰值亮度
        hdr_threshold: HDR 高光阈值
        max_long_edge: 长边最大像素数（可选）
        exposure_compensation: 曝光补偿（EV值）
        diffuse_nits: 漫反射亮度（nits）
        peak_nits: 峰值亮度（nits）
        auto_ev: 是否自动调整曝光
        auto_bright: 是否启用 rawpy 的自动亮度
        use_camera_wb: 是否使用相机白平衡
        vibrance: 自然饱和度强度（1.0=不变，1.3=默认增强）
    """
    try:
        linear_image = process_arw(arw_path, ev=ev, auto_bright=auto_bright, use_camera_wb=use_camera_wb)
    except FileNotFoundError:
        print(f"错误: ARW 文件未找到: {arw_path}")
        return False
    except Exception as e:
        print(f"处理 ARW 时出错: {e}")
        return False

    if auto_ev:
        linear_image, _ = analyze_and_adjust_exposure(linear_image, auto_adjust=True)

    if vibrance != 1.0:
        linear_image = enhance_vibrance(linear_image, strength=vibrance)

    if max_long_edge:
        linear_image = resize_image(linear_image, max_long_edge)

    if hdr_output_path:
        print("\n输出 HDR HEIF (BT.2020 + PQ)")
        try:
            bt2020_image = srgb_to_bt2020(linear_image)
            pq_image = linear_to_pq(
                bt2020_image,
                max_nits=max_nits,
                hdr_threshold=hdr_threshold,
                diffuse_nits=diffuse_nits,
                peak_nits=peak_nits,
            )
            save_heif_10bit(pq_image, hdr_output_path)
        except Exception as e:
            print(f"保存 HDR HEIF 时出错: {type(e)}, {e}")
            return False

    return True


def main():
    """
    主处理函数
    """
    parser = argparse.ArgumentParser(
        description='ARW 转 HEIF 工具 - 将 Sony ARW 文件转换为 HDR HEIF 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理说明：
  此脚本读取 Sony ARW 原始文件，应用相机白平衡，
  转换为线性 sRGB 色彩空间，然后转换为 BT.2020+PQ
  输出 HDR HEIF 格式。

  -i 参数支持文件或目录：
    文件：处理单个 ARW 文件
    目录：处理目录下所有 ARW 文件
""",
    )
    parser.add_argument('-i', '--arw', type=str, required=True, help='输入 ARW 文件或目录路径')
    parser.add_argument('-o', '--hdr-output', type=str, help='输出 HDR HEIF 文件路径')
    parser.add_argument('-m', '--max-long-edge', type=int, default=6000, help='限制长边最大像素数，超过则缩小')

    exposure_group = parser.add_argument_group('曝光控制')
    exposure_group.add_argument('--ev', type=float, default=0.0, help='手动曝光补偿（EV值），正值增加亮度')
    exposure_group.add_argument('--auto-ev', action='store_true', help='自动分析并调整曝光（推荐用于高动态范围相机）')
    exposure_group.add_argument('--auto-bright', action='store_true', help='启用 rawpy 的自动亮度调整')
    exposure_group.add_argument('--auto-wb', action='store_true', help='使用自动白平衡而非相机白平衡')

    hdr_group = parser.add_argument_group('HDR 参数（针对高动态范围相机优化）')
    hdr_group.add_argument(
        '--max-nits', type=float, default=500.0, help='HDR 峰值亮度（尼特），仅在 hdr-threshold=0 时使用'
    )
    hdr_group.add_argument(
        '--hdr-threshold', type=float, default=0.65, help='高光阈值（0-1），0 表示禁用 fake HDR。建议 A7R4: 0.6-0.7'
    )
    hdr_group.add_argument(
        '--diffuse-nits',
        type=float,
        default=700,
        help='普通白色亮度（nits），仅在 fake HDR 模式下使用。建议 A7R4: 600-800',
    )
    hdr_group.add_argument(
        '--peak-nits',
        type=float,
        default=1800.0,
        help='高光峰值亮度（nits），仅在 fake HDR 模式下使用。建议 A7R4: 1500-2000',
    )

    color_group = parser.add_argument_group('色彩控制')
    color_group.add_argument(
        '--vibrance',
        type=float,
        default=1.3,
        help='自然饱和度强度（1.0=不变，1.3=默认增强，1.5=强增强）。设为 1.0 可禁用',
    )

    args = parser.parse_args()

    print(f"输入: {args.arw}")
    print("正在查找 ARW 文件...")

    arw_files = find_arw_files(args.arw)

    if not arw_files:
        print(f"错误: 在目录 {args.arw} 中未找到 ARW 文件")
        sys.exit(1)

    print(f"找到 {len(arw_files)} 个 ARW 文件\n")

    success_count = 0
    fail_count = 0

    for i, arw_file in enumerate(arw_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(arw_files)}] 处理: {Path(arw_file).name}")
        print(f"{'='*80}")

        hdr_output_path = None

        if args.hdr_output:
            hdr_output_path = args.hdr_output
        else:
            hdr_output_path = str(Path(arw_file).with_suffix('.heif'))

        if process_single_arw(
            arw_file,
            hdr_output_path=hdr_output_path,
            max_nits=args.max_nits,
            hdr_threshold=args.hdr_threshold,
            max_long_edge=args.max_long_edge,
            ev=args.ev,
            diffuse_nits=args.diffuse_nits,
            peak_nits=args.peak_nits,
            auto_ev=args.auto_ev,
            auto_bright=args.auto_bright,
            use_camera_wb=not args.auto_wb,
            vibrance=args.vibrance,
        ):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*80}")
    print(f"批量处理完成: 成功 {success_count} 个，失败 {fail_count} 个")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
