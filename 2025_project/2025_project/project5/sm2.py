import secrets
import hashlib
from typing import Tuple, Optional

# SM2椭圆曲线参数定义
SM2_PRIME = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
SM2_COEFF_A = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
SM2_COEFF_B = 0x28E9FA9E9D9F5E344D5AEF7E8B5D50A0C648FEE9A97A7E37BBA2DDF1D5
SM2_ORDER = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
BASE_POINT_X = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
BASE_POINT_Y = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
GENERATOR = (BASE_POINT_X, BASE_POINT_Y)


def modular_inverse(value: int, modulus: int) -> int:
    """计算模逆"""
    return pow(value, -1, modulus)


def elliptic_curve_add(point1: Tuple[int, int], point2: Tuple[int, int]) -> Tuple[int, int]:
    """椭圆曲线点加法"""
    # 处理无穷远点
    if point1 == (0, 0):
        return point2
    if point2 == (0, 0):
        return point1

    x1, y1 = point1
    x2, y2 = point2

    # 处理点加法特殊情况
    if x1 == x2 and (y1 + y2) % SM2_PRIME == 0:
        return (0, 0)  # 无穷远点

    # 计算斜率
    if point1 == point2:
        # 点倍乘情况
        numerator = 3 * x1 * x1 + SM2_COEFF_A
        denominator = 2 * y1
        slope = (numerator * modular_inverse(denominator, SM2_PRIME)) % SM2_PRIME
    else:
        # 普通点加法
        numerator = y2 - y1
        denominator = x2 - x1
        slope = (numerator * modular_inverse(denominator, SM2_PRIME)) % SM2_PRIME

    # 计算结果点坐标
    result_x = (slope * slope - x1 - x2) % SM2_PRIME
    result_y = (slope * (x1 - result_x) - y1) % SM2_PRIME

    return (result_x, result_y)


def point_multiplication(scalar: int, base_point: Tuple[int, int]) -> Tuple[int, int]:
    """椭圆曲线标量乘法"""
    result = (0, 0)  # 无穷远点
    current_point = base_point

    while scalar > 0:
        if scalar & 1:
            result = elliptic_curve_add(result, current_point)
        current_point = elliptic_curve_add(current_point, current_point)
        scalar >>= 1

    return result


def generate_private_public_pair() -> Tuple[int, Tuple[int, int]]:
    """生成SM2密钥对"""
    private_key = secrets.randbelow(SM2_ORDER - 1) + 1
    public_key = point_multiplication(private_key, GENERATOR)
    return private_key, public_key


def create_digital_signature(message: bytes, private_key: int) -> Tuple[int, int]:
    """创建SM2数字签名"""
    # 计算消息哈希
    message_hash = int(hashlib.sha256(message).hexdigest(), 16)

    while True:
        # 生成随机数
        random_k = secrets.randbelow(SM2_ORDER - 1) + 1

        # 计算R点
        temp_point = point_multiplication(random_k, GENERATOR)
        r_value = (message_hash + temp_point[0]) % SM2_ORDER

        # 检查r值有效性
        if r_value == 0 or r_value + random_k == SM2_ORDER:
            continue

        # 计算s值
        temp_inv = modular_inverse(1 + private_key, SM2_ORDER)
        s_value = (temp_inv * (random_k - r_value * private_key)) % SM2_ORDER

        if s_value != 0:
            return (r_value, s_value)


def verify_digital_signature(message: bytes, signature: Tuple[int, int], public_key: Tuple[int, int]) -> bool:
    """验证SM2数字签名"""
    r_val, s_val = signature

    # 验证签名参数范围
    if not (1 <= r_val <= SM2_ORDER - 1 and 1 <= s_val <= SM2_ORDER - 1):
        return False

    # 计算消息哈希
    msg_hash = int(hashlib.sha256(message).hexdigest(), 16)

    # 计算t值
    t_value = (r_val + s_val) % SM2_ORDER
    if t_value == 0:
        return False

    # 计算验证点
    point1 = point_multiplication(s_val, GENERATOR)
    point2 = point_multiplication(t_value, public_key)
    verification_point = elliptic_curve_add(point1, point2)

    # 验证结果
    computed_r = (msg_hash + verification_point[0]) % SM2_ORDER
    return computed_r == r_val


def main():
    """主函数：演示SM2签名和验证"""
    print("=== SM2椭圆曲线数字签名算法演示 ===")

    # 生成密钥对
    secret_key, public_key = generate_private_public_pair()
    print(f"私钥: {secret_key}")
    print(f"公钥: ({public_key[0]}, {public_key[1]})")

    # 测试消息
    test_message = b"SM2"
    print(f"\n原始消息: {test_message.decode('utf-8')}")

    # 创建签名
    signature = create_digital_signature(test_message, secret_key)
    print(f"签名: (r={signature[0]}, s={signature[1]})")

    # 验证签名
    is_valid = verify_digital_signature(test_message, signature, public_key)
    print(f"签名验证结果: {'成功' if is_valid else '失败'}")

    # 测试错误消息
    wrong_message = b"126747"
    is_wrong_valid = verify_digital_signature(wrong_message, signature, public_key)
    print(f"错误消息验证结果: {'成功' if is_wrong_valid else '失败'}")


if __name__ == "__main__":
    main()
