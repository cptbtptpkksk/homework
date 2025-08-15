import cv2
import numpy as np
import argparse
import os
from scipy.fftpack import dct, idct
import math


# ---------------------------
# 工具函数
# ---------------------------
def convert_to_gray(img):
    """将图像转换为灰度图（若为彩色图）"""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def calculate_psnr(img1, img2):
    """计算两张图像的PSNR值"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# ---------------------------
# DCT处理工具（8x8块操作）
# ---------------------------
def process_blocks(img, block_size, process_func):
    """对图像进行分块处理并应用指定函数"""
    height, width = img.shape
    output = np.zeros_like(img, dtype=np.float32)
    for block_row in range(0, height, block_size):
        for block_col in range(0, width, block_size):
            # 提取当前8x8块
            block = img[block_row:block_row + block_size,
                    block_col:block_col + block_size]
            # 跳过不完整的块（确保图像尺寸为块大小的整数倍）
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue
            # 应用处理函数并放回结果
            processed_block = process_func(block)
            output[block_row:block_row + block_size,
            block_col:block_col + block_size] = processed_block
    return output


def compute_dct2(block):
    """计算二维DCT变换"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def compute_idct2(block):
    """计算二维逆DCT变换"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# ---------------------------
# 水印嵌入与提取（盲提取）
# 算法说明：
# - 将图像转换为YCrCb色彩空间，使用Y通道进行处理
# - 将二值水印调整为与8x8块数量匹配的网格尺寸
# - 对每个8x8块，选择两个中频系数位置，避免直流分量和高频分量
# - 若水印比特为1：使系数1 > 系数2 + 强度因子
#   若水印比特为0：使系数1 < 系数2 - 强度因子
# - 通过IDCT重建Y通道并合并回BGR图像
# 提取过程：
# - 相同块划分，计算DCT后通过系数差值符号判断水印比特
# 实现盲提取（无需原始载体图像）
# ---------------------------

# 选择8x8块中的中频系数位置（避免(0,0)直流分量）
# 常用选择：(2,1)和(1,2)或(3,2)和(2,3)
MID_FREQ_POS1 = (2, 1)  # 第一个中频系数位置
MID_FREQ_POS2 = (1, 2)  # 第二个中频系数位置
BLOCK_SIZE = 8  # 块大小固定为8x8


def embed_dct_watermark(cover_img, watermark_bin, strength=10.0):
    """
    嵌入水印到载体图像

    参数:
    cover_img: BGR格式的载体图像（uint8）
    watermark_bin: 二值水印数组（0/1值）
    strength: 水印嵌入强度

    返回:
    watermarked_img: 含水印图像（uint8）
    """
    # 确保载体图像尺寸为块大小的整数倍
    height, width = cover_img.shape[:2]
    adj_height = (height // BLOCK_SIZE) * BLOCK_SIZE
    adj_width = (width // BLOCK_SIZE) * BLOCK_SIZE
    cover = cover_img[:adj_height, :adj_width].copy()

    # 转换到YCrCb空间并提取Y通道
    ycrcb_img = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_channel = ycrcb_img[:, :, 0]

    # 计算块数量
    block_rows = adj_height // BLOCK_SIZE
    block_cols = adj_width // BLOCK_SIZE

    # 调整水印尺寸以匹配块网格
    wm_height, wm_width = watermark_bin.shape
    wm_resized = cv2.resize((watermark_bin.astype(np.uint8) * 255),
                            (block_cols, block_rows),
                            interpolation=cv2.INTER_NEAREST)
    wm_bits = (wm_resized > 127).astype(np.uint8)

    # 定义块处理函数
    def process_block(block):
        # 转换为浮点型并计算DCT
        block_float = block.astype(np.float32)
        dct_coeffs = compute_dct2(block_float)

        # 获取当前块对应的水印比特
        b = wm_bits[process_block.current_row, process_block.current_col]

        # 调整DCT系数
        c1 = dct_coeffs[MID_FREQ_POS1]
        c2 = dct_coeffs[MID_FREQ_POS2]
        if b == 1:
            if c1 <= c2 + strength:
                dct_coeffs[MID_FREQ_POS1] = c2 + strength + 1.0
        else:
            if c1 >= c2 - strength:
                dct_coeffs[MID_FREQ_POS1] = c2 - strength - 1.0

        # 逆DCT变换返回
        return compute_idct2(dct_coeffs)

    # 遍历所有块并应用处理
    watermarked_y = np.zeros_like(y_channel, dtype=np.float32)
    for row in range(block_rows):
        for col in range(block_cols):
            # 记录当前块位置
            process_block.current_row = row
            process_block.current_col = col
            # 提取并处理当前块
            y_block = y_channel[row * BLOCK_SIZE:(row + 1) * BLOCK_SIZE,
                      col * BLOCK_SIZE:(col + 1) * BLOCK_SIZE]
            processed_block = process_block(y_block)
            watermarked_y[row * BLOCK_SIZE:(row + 1) * BLOCK_SIZE,
            col * BLOCK_SIZE:(col + 1) * BLOCK_SIZE] = processed_block

    # 整合结果并转换回BGR
    ycrcb_img[:, :, 0] = np.clip(watermarked_y, 0, 255)
    watermarked_img = cv2.cvtColor(ycrcb_img.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return watermarked_img, wm_bits


def extract_dct_watermark(watermarked_img, wm_shape):
    """
    从含水印图像中提取水印

    参数:
    watermarked_img: 含水印图像（BGR格式，uint8）
    wm_shape: 目标水印尺寸 (高度, 宽度)

    返回:
    extracted_bits: 提取的二值水印（0/1数组）
    """
    height, width = watermarked_img.shape[:2]
    adj_height = (height // BLOCK_SIZE) * BLOCK_SIZE
    adj_width = (width // BLOCK_SIZE) * BLOCK_SIZE
    img = watermarked_img[:adj_height, :adj_width].copy()

    # 提取Y通道
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_channel = ycrcb_img[:, :, 0]

    # 计算块数量
    block_rows = adj_height // BLOCK_SIZE
    block_cols = adj_width // BLOCK_SIZE

    # 提取水印比特
    extracted_bits = np.zeros((block_rows, block_cols), dtype=np.uint8)
    for row in range(block_rows):
        for col in range(block_cols):
            # 提取块并计算DCT
            y_block = y_channel[row * BLOCK_SIZE:(row + 1) * BLOCK_SIZE,
                      col * BLOCK_SIZE:(col + 1) * BLOCK_SIZE]
            dct_coeffs = compute_dct2(y_block.astype(np.float32))
            # 判断比特值
            c1 = dct_coeffs[MID_FREQ_POS1]
            c2 = dct_coeffs[MID_FREQ_POS2]
            extracted_bits[row, col] = 1 if (c1 - c2) > 0 else 0

    # 调整为目标水印尺寸
    extracted_resized = cv2.resize((extracted_bits.astype(np.uint8) * 255),
                                   (wm_shape[1], wm_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
    return (extracted_resized > 127).astype(np.uint8)


# ---------------------------
# 攻击函数
# ---------------------------
def flip_attack(img, direction='horizontal'):
    """图像翻转攻击"""
    if direction == 'horizontal':
        return cv2.flip(img, 1)
    elif direction == 'vertical':
        return cv2.flip(img, 0)
    return img.copy()


def translate_attack(img, tx=10, ty=5):
    """图像平移攻击"""
    height, width = img.shape[:2]
    # 平移矩阵
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, trans_mat, (width, height), borderMode=cv2.BORDER_REFLECT)


def crop_attack(img, crop_ratio=0.8):
    """图像裁剪攻击（中心裁剪后填充回原尺寸）"""
    height, width = img.shape[:2]
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)
    # 计算裁剪区域
    y_start = (height - crop_height) // 2
    x_start = (width - crop_width) // 2
    cropped = img[y_start:y_start + crop_height, x_start:x_start + crop_width]
    # 填充回原尺寸
    top = y_start
    bottom = height - (y_start + crop_height)
    left = x_start
    right = width - (x_start + crop_width)
    return cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_REFLECT)


def contrast_attack(img, factor=1.2, offset=0):
    """对比度调整攻击"""
    adjusted = img.astype(np.float32) * factor + offset
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def noise_attack(img, noise_level=5.0):
    """高斯噪声攻击"""
    noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def jpeg_attack(img, quality=50):
    """JPEG压缩攻击"""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_params)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


# ---------------------------
# 评估函数
# ---------------------------
def calculate_accuracy(original_wm, extracted_wm):
    """计算水印提取准确率"""
    total_bits = original_wm.size
    matching_bits = np.sum(original_wm == extracted_wm)
    return float(matching_bits) / float(total_bits)


def run_evaluation(cover_path, wm_path, output_dir='output', strength=10.0):
    """运行水印嵌入、提取与鲁棒性评估流程"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取输入图像
    cover_img = cv2.imread(cover_path)
    if cover_img is None:
        raise ValueError("无法读取载体图像")
    wm_gray = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    if wm_gray is None:
        raise ValueError("无法读取水印图像")

    # 水印二值化
    _, wm_binary = cv2.threshold(wm_gray, 127, 1, cv2.THRESH_BINARY)

    # 嵌入水印
    watermarked_img, wm_grid = embed_dct_watermark(cover_img, wm_binary, strength=strength)
    output_path = os.path.join(output_dir, 'watermarked.png')
    cv2.imwrite(output_path, watermarked_img)
    print(f"已保存含水印图像至: {output_path}")

    # 无攻击提取
    extracted_clean = extract_dct_watermark(watermarked_img, wm_binary.shape)
    cv2.imwrite(os.path.join(output_dir, 'extracted_clean.png'), extracted_clean * 255)
    acc_clean = calculate_accuracy(wm_binary, extracted_clean)
    psnr_value = calculate_psnr(cover_img[:watermarked_img.shape[0], :watermarked_img.shape[1]], watermarked_img)
    print(f"无攻击提取准确率: {acc_clean * 100:.2f}%  载体与含水印图像PSNR: {psnr_value:.2f} dB")

    # 执行攻击测试
    attack_list = [
        ('水平翻转', lambda x: flip_attack(x, 'horizontal')),
        ('垂直翻转', lambda x: flip_attack(x, 'vertical')),
        ('平移', lambda x: translate_attack(x, tx=6, ty=4)),
        ('80%裁剪', lambda x: crop_attack(x, 0.8)),
        ('高对比度', lambda x: contrast_attack(x, factor=1.3)),
        ('低对比度', lambda x: contrast_attack(x, factor=0.7)),
        ('高斯噪声', lambda x: noise_attack(x, noise_level=5.0)),
        ('JPEG(Q70)', lambda x: jpeg_attack(x, quality=70)),
        ('JPEG(Q50)', lambda x: jpeg_attack(x, quality=50)),
    ]

    results = []
    for attack_name, attack_func in attack_list:
        # 执行攻击
        attacked_img = attack_func(watermarked_img)
        attacked_path = os.path.join(output_dir, f'attacked_{attack_name}.png')
        cv2.imwrite(attacked_path, attacked_img)

        # 提取水印
        extracted = extract_dct_watermark(attacked_img, wm_binary.shape)
        extract_path = os.path.join(output_dir, f'extracted_{attack_name}.png')
        cv2.imwrite(extract_path, extracted * 255)

        # 计算准确率
        accuracy = calculate_accuracy(wm_binary, extracted)
        print(f"{attack_name}攻击后提取准确率: {accuracy * 100:.2f}%  已保存攻击图像与提取结果")
        results.append((attack_name, accuracy))

    # 输出结果 summary
    print("\n评估结果汇总:")
    for name, acc in results:
        print(f"{name:10s} 准确率: {acc * 100:6.2f}%")
    return results


# ---------------------------
# 命令行接口
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于DCT的盲水印算法演示")
    parser.add_argument('--cover', required=True, help="载体图像路径")
    parser.add_argument('--watermark', required=True, help="二值水印图像路径（建议黑白图）")
    parser.add_argument('--out', default='output', help="结果输出目录")
    parser.add_argument('--strength', type=float, default=10.0, help="水印嵌入强度（默认10.0）")
    args = parser.parse_args()

    run_evaluation(args.cover, args.watermark, output_dir=args.out, strength=args.strength)

