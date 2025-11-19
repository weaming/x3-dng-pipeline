# -*- coding: utf-8 -*-

"""
DNG Color Profile (DCP) Look 应用工具

功能：
- 从 DCP XML 文件解析色彩风格配置
- 处理已预处理为线性 sRGB 的 DNG 文件
  （由修改版 x3f tools 导出，已应用白平衡）
- 应用 DCP 色彩风格调整
- 输出专业级 sRGB JPEG 成片

处理流程：
1. 读取 DNG（线性 sRGB 空间，白平衡已应用）
2. 应用 HueSatDeltas LUT（色相、饱和度、亮度调整）
3. 应用 Tone Curve（对比度和亮度映射）
4. 应用 sRGB Gamma 校正

注意：
- 此脚本专为已预处理的 DNG 设计
- DNG 中的 ColorMatrix 仅供参考，不参与处理
- 主要应用 DCP 中的"look"（HueSatDeltas + ToneCurve）
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rawpy
from scipy.interpolate import RegularGridInterpolator, interp1d


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
        input_values,
        output_values,
        kind='cubic',
        bounds_error=False,
        fill_value=(output_values[0], output_values[-1])
    )

    adjusted = tone_func(image_rgb).astype(np.float32)

    return np.clip(adjusted, 0, 1)


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


def srgb_to_linear(srgb):
    """
    将 sRGB 转换为线性 RGB，移除 Gamma 编码
    """
    linear = np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))

    return linear


def process_dng_with_dcp(dng_path, dcp_data):
    """
    使用 DCP 配置处理 DNG 文件的完整流程

    注意：此脚本处理的是已预处理为线性 sRGB 的 DNG 文件
    （由修改版 x3f tools 导出，已应用白平衡）

    流程：
    1. 读取 DNG（已是线性 sRGB 空间）
    2. 在 HSV 空间应用 HueSatDeltas LUT（色彩风格调整）
    3. 应用 Tone Curve（亮度/对比度调整）
    4. 应用 sRGB Gamma 校正
    """
    print(f"\n读取文件: {dng_path}")

    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.sRGB
        )

    rgb_float = rgb.astype(np.float32) / 65535.0

    processed = rgb_float

    if dcp_data.get('hue_sat_lut') is not None:
        processed = apply_hsv_lut(processed, dcp_data['hue_sat_lut'])
    else:
        print("警告: 未找到 HueSatDeltas LUT，跳过色彩风格调整")

    if dcp_data.get('tone_curve') is not None:
        processed = apply_tone_curve(processed, dcp_data['tone_curve'])
    else:
        print("警告: 未找到 Tone Curve，跳过对比度调整")

    final_rgb = linear_to_srgb(processed)

    return final_rgb


def main():
    """
    主处理函数
    """
    parser = argparse.ArgumentParser(
        description='DCP Look 应用工具 - 处理已预处理的 DNG 并应用色彩风格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理说明：
  此脚本专为由修改版 x3f tools 导出的 DNG 设计，这些 DNG：
  - 已转换为线性 sRGB 色彩空间
  - 已应用白平衡增益
  - ColorMatrix 仅供参考，不参与处理

  脚本会读取 DNG 的线性 sRGB 数据，应用 DCP 中的色彩风格
  （HueSatDeltas LUT 和 Tone Curve），然后应用 sRGB Gamma
  校正输出 JPEG 成片。
"""
    )
    parser.add_argument('--dng', type=str, default='files/色卡阴天.dng', help='输入 DNG 文件路径')
    parser.add_argument('--xml', type=str, default='dp3q/DP3Q.xml', help='输入 DCP XML 配置文件路径')
    parser.add_argument('--output', type=str, help='输出 JPG 文件路径（默认：<dng>.jpg）')
    parser.add_argument('--quality', type=int, default=100, help='输出 JPG 质量 (0-100，默认：100)')

    args = parser.parse_args()
    if not args.output:
        args.output = args.dng + '.jpg'

    print(f"输入 DNG: {args.dng}")
    print(f"DCP XML: {args.xml}")
    print(f"输出 JPG: {args.output}")

    try:
        dcp_data = parse_dcp_xml(args.xml)
    except FileNotFoundError:
        print(f"错误: XML 配置文件未找到: {args.xml}")
        sys.exit(1)
    except Exception as e:
        print(f"解析 XML 时出错: {e}")
        sys.exit(1)

    try:
        final_image = process_dng_with_dcp(args.dng, dcp_data)
    except FileNotFoundError:
        print(f"错误: DNG 文件未找到: {args.dng}")
        sys.exit(1)
    except Exception as e:
        print(f"处理 DNG 时出错: {e}")
        sys.exit(1)

    output_image_8bit = np.clip(final_image * 255.0, 0, 255).astype(np.uint8)
    output_image_bgr = cv2.cvtColor(output_image_8bit, cv2.COLOR_RGB2BGR)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n保存 JPG 文件到: {args.output} (质量: {args.quality})")
    try:
        cv2.imwrite(args.output, output_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        print("✓ 处理完成！")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
