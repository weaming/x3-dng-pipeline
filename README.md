# X3F 色卡校正工具

使用 ColorChecker Passport Photo 2 色卡对 Sigma X3F RAW 文件进行色彩校正。

1. 使用 [weaming/x3f-go](https://github.com/weaming/x3f-go) 导出 DNG
2. `python3 apply_dcp.py -i path/to/dng/file/or/directory`

```
usage: apply_dcp.py [-h] [-i DNG] [--xml XML] [--output OUTPUT] [-o HDR_OUTPUT] [-q QUALITY] [--max-nits MAX_NITS] [--hdr-threshold HDR_THRESHOLD]

DCP 应用工具 - 处理已预处理的 DNG 并应用色彩风格

options:
  -h, --help            show this help message and exit
  -i, --dng DNG         输入 DNG 文件或目录路径
  --xml XML             输入 DCP XML 配置文件路径
  --output OUTPUT       输出 JPG 文件路径
  -o, --hdr-output HDR_OUTPUT
                        输出 HDR HEIF 文件路径
  -q, --quality QUALITY
                        JPG 输出质量 (0-100)
  --max-nits MAX_NITS   HDR 峰值亮度（尼特）
  --hdr-threshold HDR_THRESHOLD
                        超过多少亮度的像素开始被视为高光

处理说明：
  此脚本专为由 x3f-go 导出的 DNG 设计，这些 DNG：
  - 已转换为线性 sRGB 色彩空间
  - 已应用白平衡增益

  脚本会读取 DNG 的线性 sRGB 数据，
  应用 DCP 中的校色数据（HueSatDeltas LUT 和 Tone Curve），
  然后应用 sRGB Gamma 校正输出 JPEG，
  或者转化为 BT2020+PQ 输出 HEIF。

  --dng 参数支持文件或目录：
    文件：处理单个 DNG 文件
    目录：处理目录下所有 DNG 文件（仅一级）
```
