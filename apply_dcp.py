#!/usr/bin/env python3

"""
DNG Color Profile (DCP) Look 应用工具

功能：
- 从 DCP XML 文件解析色彩风格配置
- 处理已预处理为线性 sRGB 的 DNG 文件
  （由修改版 x3f tools 导出，已应用白平衡）
- 应用 DCP 色彩风格调整
- 输出专业级 sRGB JPEG 成片（SDR）
- 支持输出 PQ HDR HEIF 格式（10-bit）

处理流程：
1. 读取 DNG（线性 sRGB 空间，白平衡已应用）
2. 应用 HueSatDeltas LUT（色相、饱和度、亮度调整）
3. 应用 Tone Curve（对比度和亮度映射）
4. 输出分支：
   - SDR: 应用 sRGB Gamma 校正 → 8-bit JPEG
   - HDR: 色域转换 (sRGB→BT.2020) → PQ 编码 → HEIF

HDR 输出说明（符合 HDR10 标准）：
- 色域：BT.2020 (ITU-R BT.2020)
- 传输函数：PQ (SMPTE ST 2084)
- 支持自定义峰值亮度（默认 1000 nits）
- HEIF 容器，高压缩率

注意：
- 此脚本专为已预处理的 DNG 设计
- DNG 中的 ColorMatrix 仅供参考，不参与处理
- 主要应用 DCP 中的"look"（HueSatDeltas + ToneCurve）
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import rawpy
from pillow_heif import register_heif_opener
from scipy.interpolate import RegularGridInterpolator, interp1d

register_heif_opener()
DEBUG = bool(os.getenv('DEBUG'))


def parse_dcp_xml(xml_path):
    """
    解析 DCP XML 文件，提取色彩风格信息

    注意：此脚本处理的 DNG 已是线性 sRGB，因此只提取色彩风格
    （HueSatDeltas LUT 和 Tone Curve），ColorMatrix 不参与处理

    Returns:
        dict: 包含 HueSatDeltas LUT 和 Tone Curve 的字典
    """
    print(f"解析 DCP 配置文件: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    dcp_data = {}

    hsd_node = root.find('HueSatDeltas1')
    if hsd_node is not None:
        hue_divs = int(hsd_node.get('hueDivisions'))
        sat_divs = int(hsd_node.get('satDivisions'))

        lut = np.zeros((hue_divs, sat_divs, 3), dtype=np.float32)
        lut[:, :, 1] = 1.0
        lut[:, :, 2] = 1.0

        for element in hsd_node:
            h = int(element.get('HueDiv'))
            s = int(element.get('SatDiv'))
            h_shift = float(element.get('HueShift'))
            s_scale = float(element.get('SatScale'))
            v_scale = float(element.get('ValScale'))
            if h < hue_divs and s < sat_divs:
                lut[h, s, 0] = h_shift
                lut[h, s, 1] = s_scale
                lut[h, s, 2] = v_scale

        dcp_data['hue_sat_lut'] = lut
        print(f"已加载 HueSatDeltas LUT: {hue_divs}x{sat_divs}")
    else:
        print("警告: 未找到 HueSatDeltas1")
        dcp_data['hue_sat_lut'] = None

    tone_curve_node = root.find('ToneCurve')
    if tone_curve_node is not None:
        size = int(tone_curve_node.get('Size'))
        curve_points = []
        for element in tone_curve_node:
            h = float(element.get('h'))
            v = float(element.get('v'))
            curve_points.append((h, v))

        curve_points.sort(key=lambda x: x[0])
        dcp_data['tone_curve'] = np.array(curve_points, dtype=np.float32)
        print(f"已加载 ToneCurve: {size} 个控制点")
    else:
        print("警告: 未找到 ToneCurve")
        dcp_data['tone_curve'] = None

    return dcp_data


def apply_hsv_lut(image_rgb_float, lut):
    """
    在 HSV 空间应用完整的 HueSatDeltas 查找表
    包括 HueShift、SatScale 和 ValScale 三个维度的调整
    """
    if lut is None:
        return image_rgb_float

    print("应用 HueSatDeltas 查找表（包含色相、饱和度、亮度调整）...")

    hue_divs, sat_divs, _ = lut.shape

    h_coords = np.linspace(0, 360, hue_divs)
    s_coords = np.linspace(0, 1, sat_divs)

    interp_hue_shift = RegularGridInterpolator(
        (h_coords, s_coords), lut[:, :, 0], method='linear', bounds_error=False, fill_value=0
    )
    interp_sat_scale = RegularGridInterpolator(
        (h_coords, s_coords), lut[:, :, 1], method='linear', bounds_error=False, fill_value=1
    )
    interp_val_scale = RegularGridInterpolator(
        (h_coords, s_coords), lut[:, :, 2], method='linear', bounds_error=False, fill_value=1
    )

    hsv_image = cv2.cvtColor(image_rgb_float, cv2.COLOR_RGB2HSV)

    h_flat = hsv_image[:, :, 0].flatten()
    s_flat = hsv_image[:, :, 1].flatten()
    query_points = np.vstack((h_flat, s_flat)).T

    hue_shifts = interp_hue_shift(query_points).reshape(hsv_image.shape[:2])
    sat_scales = interp_sat_scale(query_points).reshape(hsv_image.shape[:2])
    val_scales = interp_val_scale(query_points).reshape(hsv_image.shape[:2])

    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shifts) % 360
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * sat_scales, 0, 1.0)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * val_scales, 0, 1.0)

    adjusted_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return adjusted_rgb


def apply_tone_curve(image_rgb, tone_curve):
    """
    应用 Tone Curve 进行亮度/对比度调整

    tone_curve: Nx2 数组，第一列是输入值 (h)，第二列是输出值 (v)
    """
    if tone_curve is None:
        return image_rgb

    print("应用 Tone Curve 调整...")

    input_values = tone_curve[:, 0]
    output_values = tone_curve[:, 1]

    tone_func = interp1d(
        input_values, output_values, kind='cubic', bounds_error=False, fill_value=(output_values[0], output_values[-1])
    )

    adjusted = tone_func(image_rgb).astype(np.float32)

    return np.clip(adjusted, 0, 1)


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


def find_dng_files(directory):
    """
    遍历目录，查找所有 DNG 文件（仅一级子目录）

    Args:
        directory: 目录路径

    Returns:
        DNG 文件路径列表
    """
    dng_files = []
    dir_path = Path(directory)

    if not dir_path.is_dir():
        dng_files.append(str(dir_path))
        return dng_files

    if not dir_path.is_dir():
        return []

    for file_path in dir_path.glob('*.dng'):
        if file_path.is_file():
            dng_files.append(str(file_path))

    for file_path in dir_path.glob('*.DNG'):
        if file_path.is_file() and str(file_path) not in dng_files:
            dng_files.append(str(file_path))

    dng_files.sort()
    return dng_files


def process_dng_with_dcp(dng_path, dcp_data):
    """
    使用 DCP 配置处理 DNG 文件的完整流程

    注意：此脚本处理的是已预处理为线性 sRGB 的 DNG 文件
    （由修改版 x3f tools 导出，已应用白平衡）

    流程：
    1. 读取 DNG（已是线性 sRGB 空间）
    2. 在 HSV 空间应用 HueSatDeltas LUT（色彩风格调整）
    3. 应用 Tone Curve（亮度/对比度调整）

    Returns:
        处理后的线性 RGB 图像（未应用 gamma/PQ 编码）
    """
    print(f"\n读取文件: {dng_path}")

    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.sRGB,
        )

    rgb_float = rgb.astype(np.float32) / 65535.0
    if DEBUG:
        stat_image(rgb_float)

    processed = rgb_float

    if dcp_data.get('hue_sat_lut') is not None:
        processed = apply_hsv_lut(processed, dcp_data['hue_sat_lut'])
    else:
        print("警告: 未找到 HueSatDeltas LUT，跳过色彩风格调整")

    if dcp_data.get('tone_curve') is not None:
        processed = apply_tone_curve(processed, dcp_data['tone_curve'])
    else:
        print("警告: 未找到 Tone Curve，跳过对比度调整")

    return processed


def linear_to_srgb(linear_rgb):
    """
    将线性 RGB 转换为 sRGB，应用标准 Gamma 校正

    sRGB gamma 函数：
    - linear <= 0.0031308: srgb = 12.92 * linear
    - linear > 0.0031308: srgb = 1.055 * linear^(1/2.4) - 0.055
    """
    print("应用 sRGB Gamma 校正...")

    srgb = np.where(
        linear_rgb <= 0.0031308, 12.92 * linear_rgb, 1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055
    ).astype(np.float32)

    return np.clip(srgb, 0, 1)


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


def linear_to_pq(bt2020_image, *, max_nits=1000.0, hdr_threshold=0.0):
    """
    将线性 RGB 转换为 PQ (Perceptual Quantizer) HDR

    PQ 是 SMPTE ST 2084 / ITU-R BT.2100 标准定义的 EOTF
    用于 HDR10 内容

    Args:
        bt2020_image: 线性光值（BT.2020 色域），范围 [0, inf)
        max_nits: 内容的最大亮度（尼特），默认 1000 nits
                  常见值：1000（消费级HDR）, 4000（专业级）, 10000（参考级）

    Returns:
        PQ 编码值，范围 [0, 1]
    """
    print(f"应用 PQ HDR 编码（目标峰值亮度: {max_nits} nits）...")

    pq_max_nits = 10000.0
    if hdr_threshold > 0:
        fake_nits = apply_fake_hdr_effect(bt2020_image, diffuse_nits=300, peak_nits=1500, threshold=hdr_threshold)
        normalized = fake_nits / pq_max_nits
    else:
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
    # 1. 基础映射：先把所有东西都映射到漫反射亮度
    # 此时 1.0 变成了 203 nits
    nits_img = linear_img * diffuse_nits

    # 2. 提取亮度信息 (Y) 用于遮罩
    # 使用 Rec.2020 的亮度系数
    luminance = 0.2627 * linear_img[:, :, 0] + 0.6780 * linear_img[:, :, 1] + 0.0593 * linear_img[:, :, 2]

    # 3. 制作高光遮罩 (Mask)
    # 只有亮度超过 threshold 的地方才会被额外提亮
    # 使用平滑过渡 (Smoothstep) 避免出现明显的亮度断层
    mask = np.clip((luminance - threshold) / (1.0 - threshold), 0, 1)
    # 让遮罩的曲线更陡峭，只影响极亮部分
    mask = np.power(mask, 2)

    # 4. 计算额外的增益
    # 我们希望原本 1.0 的地方，最终能达到 peak_nits
    # 当前它只有 diffuse_nits。所以需要额外的增益量。
    extra_boost_factor = peak_nits / diffuse_nits

    # 5. 应用增益
    # 也就是：基础亮度 + (基础亮度 * 遮罩 * 额外增益系数)
    # 只有 mask > 0 的地方（高光）才会变亮
    # 这里的 expand_dims 是为了让 mask (H,W) 能乘 (H,W,3)
    final_nits = nits_img + (nits_img * mask[:, :, np.newaxis] * (extra_boost_factor - 1))
    return final_nits


def save_heif_10bit(image_float, output_path):
    """
    保存 HDR10 HEIF 图像

    Args:
        image_float: PQ 编码的图像数据
        output_path: 输出路径
        quality: 压缩质量
                 - 90 (默认): 高质量，类似 iPhone（推荐）
                 - 95-98: 极高质量，文件略大
                 - -1: 无损压缩，文件很大
    """
    print(f"保存 HEIF 文件到: {output_path}")
    print(f"输入图像形状: {image_float.shape}, 数据类型: {image_float.dtype}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # pillow_heif/libheif 在接收 'RGB;16' 输入时，通常期望它是占满 uint16 范围的数据。
    pq_int16 = (image_float * 65535.0).clip(0, 65535).astype(np.uint16)

    # 获取宽高
    height, width, channels = pq_int16.shape

    # Pillow 不支持 16-bit RGB 模式，所以我们直接把 numpy 字节流喂给 pillow_heif
    heif_file = pillow_heif.from_bytes(
        mode="RGB;16",  # 告诉它这是 16位 RGB 数据
        size=(width, height),  # 注意顺序是 (宽, 高)
        data=pq_int16.tobytes(),  # 转为二进制流
    )

    # 定义 NCLX (HDR 必须)
    nclx_profile = {
        "color_primaries": 9,  # BT.2020
        "transfer_characteristics": 16,  # PQ
        "matrix_coefficients": 9,  # BT.2020 Non-constant
        "full_range_flag": True,
    }
    encoder_params = {"chroma": "444"}

    heif_file.save(output_path, quality=-1, bit_depth=10, nclx_profile=nclx_profile, enc_params=encoder_params)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ HEIF 文件保存完成: {output_path} ({file_size_mb:.2f} MB)")


def process_single_dng(
    dng_path,
    dcp_data,
    output_path=None,
    hdr_output_path=None,
    quality=100,
    max_nits=203.0,
    hdr_threshold=0.6,
):
    """
    处理单个 DNG 文件

    Args:
        dng_path: DNG 文件路径
        dcp_data: 解析后的 DCP 数据
        output_path: SDR JPEG 输出路径（可选）
        hdr_output_path: HDR HEIF 输出路径（可选）
        quality: 输出质量
        max_nits: HDR 峰值亮度
        hdr_threshold: HDR 高光阈值
    """
    try:
        linear_image = process_dng_with_dcp(dng_path, dcp_data)
    except FileNotFoundError:
        print(f"错误: DNG 文件未找到: {dng_path}")
        return False
    except Exception as e:
        print(f"处理 DNG 时出错: {e}")
        return False

    if hdr_output_path:
        print("输出 HDR HEIF (BT.2020 + PQ)")
        try:
            bt2020_image = srgb_to_bt2020(linear_image)
            pq_image = linear_to_pq(bt2020_image, max_nits=max_nits, hdr_threshold=hdr_threshold)
            save_heif_10bit(pq_image, hdr_output_path)
        except Exception as e:
            print(f"保存 HDR HEIF 时出错: {type(e)}, {e}")
            return False

    if output_path:
        print("\n=== 输出 SDR JPEG ===")
        try:
            sdr_image = linear_to_srgb(linear_image)
            output_image_8bit = np.clip(sdr_image * 255.0, 0, 255).astype(np.uint8)
            output_image_bgr = cv2.cvtColor(output_image_8bit, cv2.COLOR_RGB2BGR)

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            print(f"保存 JPG 文件到: {output_path} (质量: {quality})")
            cv2.imwrite(output_path, output_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            print("✓ JPG 保存完成")
        except Exception as e:
            print(f"保存 JPG 时出错: {e}")
            return False

    return True


def main():
    """
    主处理函数
    """
    parser = argparse.ArgumentParser(
        description='DCP 应用工具 - 处理已预处理的 DNG 并应用色彩风格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理说明：
  此脚本专为由 x3f-go 导出的 DNG 设计，这些 DNG：
  - 已转换为线性 sRGB 色彩空间
  - 已应用白平衡增益

  脚本会读取 DNG 的线性 sRGB 数据，
  应用 DCP 中的色彩风格（HueSatDeltas LUT 和 Tone Curve），
  然后应用 sRGB Gamma 校正输出 JPEG，
  或者转化为 BT2020+PQ 输出 HEIF。

  --dng 参数支持文件或目录：
  - 文件：处理单个 DNG 文件
  - 目录：处理目录下所有 DNG 文件（仅一级）
""",
    )
    parser.add_argument('-i', '--dng', type=str, help='输入 DNG 文件或目录路径')
    parser.add_argument('--xml', type=str, default='dp3q/DP3Q.xml', help='输入 DCP XML 配置文件路径')
    parser.add_argument('--output', type=str, help='输出 JPG 文件路径（目录时自动生成）')
    parser.add_argument('-o', '--hdr-output', type=str, help='输出 HDR HEIF 文件路径（目录时自动生成）')
    parser.add_argument('-q', '--quality', type=int, default=100, help='JPG 输出质量 (0-100)')
    parser.add_argument('--max-nits', type=float, default=203.0, help='HDR 峰值亮度（尼特）')
    parser.add_argument('--hdr-threshold', type=float, default=0.7, help='超过多少亮度的像素开始被视为高光')

    args = parser.parse_args()

    if not args.dng:
        print('missing the -i / --dng param')
        sys.exit(1)

    parent_dir = Path(__file__).parent
    try:
        xml_path = args.xml
        if not os.path.isabs(xml_path):
            xml_path = parent_dir / xml_path
        dcp_data = parse_dcp_xml(xml_path)
    except FileNotFoundError:
        print(f"错误: XML 配置文件未找到: {args.xml}")
        sys.exit(1)
    except Exception as e:
        print(f"解析 XML 时出错: {e}")
        sys.exit(1)

    print(f"输入: {args.dng}")
    print("正在查找 DNG 文件...")

    dng_files = find_dng_files(args.dng)

    if not dng_files:
        print(f"错误: 在目录 {args.dng} 中未找到 DNG 文件")
        sys.exit(1)

    print(f"找到 {len(dng_files)} 个 DNG 文件\n")

    success_count = 0
    fail_count = 0

    for i, dng_file in enumerate(dng_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(dng_files)}] 处理: {Path(dng_file).name}")
        print(f"{'='*80}")

        output_path = None
        hdr_output_path = None

        if args.output:
            output_path = args.output
        if args.hdr_output:
            hdr_output_path = args.hdr_output
        if not (args.output or args.hdr_output):
            hdr_output_path = str(Path(dng_file).with_suffix('.heif'))

        if process_single_dng(
            dng_file,
            dcp_data,
            output_path=output_path,
            hdr_output_path=hdr_output_path,
            quality=args.quality,
            max_nits=args.max_nits,
            hdr_threshold=args.hdr_threshold,
        ):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*80}")
    print(f"批量处理完成: 成功 {success_count} 个，失败 {fail_count} 个")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
