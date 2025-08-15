# 数字签名伪造攻击演示
import hashlib
import secrets
from typing import Tuple, Optional

# 椭圆曲线参数 (使用不同的曲线参数)
CURVE_PRIME = 2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1
CURVE_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
CURVE_A = 0
CURVE_B = 7
BASE_X = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
BASE_Y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
GENERATOR_POINT = (BASE_X, BASE_Y)


class WeakSignatureSystem:
    """存在安全漏洞的数字签名系统"""

    def __init__(self):
        self.field_prime = CURVE_PRIME
        self.group_order = CURVE_ORDER
        self.base_point = GENERATOR_POINT
        self.last_nonce = None  # 存储上次使用的随机数（安全漏洞）

    def compute_modular_inverse(self, value: int, modulus: int) -> int:
        """计算模逆元"""
        return pow(value, -1, modulus)

    def perform_point_addition(self, point_a: Optional[Tuple[int, int]],
                               point_b: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """执行椭圆曲线点加法"""
        # 处理无穷远点
        if point_a is None:
            return point_b
        if point_b is None:
            return point_a

        x1, y1 = point_a
        x2, y2 = point_b

        # 处理特殊情况
        if x1 == x2 and y1 != y2:
            return None  # 无穷远点

        # 计算斜率
        if point_a == point_b:
            # 点倍乘
            slope_numerator = 3 * x1 * x1
            slope_denominator = 2 * y1
        else:
            # 普通点加法
            slope_numerator = y2 - y1
            slope_denominator = x2 - x1

        slope = (slope_numerator * self.compute_modular_inverse(slope_denominator, self.field_prime)) % self.field_prime

        # 计算结果点
        result_x = (slope * slope - x1 - x2) % self.field_prime
        result_y = (slope * (x1 - result_x) - y1) % self.field_prime

        return (result_x, result_y)

    def scalar_point_multiplication(self, scalar: int, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """标量点乘法"""
        accumulator = None
        current_point = point

        while scalar > 0:
            if scalar & 1:
                accumulator = self.perform_point_addition(accumulator, current_point)
            current_point = self.perform_point_addition(current_point, current_point)
            scalar >>= 1

        return accumulator

    def create_key_pair(self) -> Tuple[int, Tuple[int, int]]:
        """生成密钥对"""
        secret_key = secrets.randbelow(self.group_order - 1) + 1
        public_key = self.scalar_point_multiplication(secret_key, self.base_point)
        return secret_key, public_key

    def generate_signature(self, message: bytes, secret_key: int,
                           nonce: Optional[int] = None) -> Tuple[int, int]:
        """生成数字签名（存在安全漏洞）"""
        # 计算消息哈希
        message_digest = int.from_bytes(hashlib.sha256(message).digest(), 'big')

        # 安全漏洞：允许指定或重用随机数
        if nonce is not None:
            k = nonce
        else:
            k = secrets.randbelow(self.group_order - 1) + 1

        self.last_nonce = k  # 存储随机数（另一个安全漏洞）

        # 计算 R = k * G
        r_point = self.scalar_point_multiplication(k, self.base_point)
        r_value = r_point[0] % self.group_order

        # 计算 s = k^(-1) * (hash + r * private_key) mod n
        k_inverse = self.compute_modular_inverse(k, self.group_order)
        s_value = (k_inverse * (message_digest + r_value * secret_key)) % self.group_order

        return (r_value, s_value)

    def validate_signature(self, message: bytes, signature: Tuple[int, int],
                           public_key: Tuple[int, int]) -> bool:
        """验证数字签名"""
        r_val, s_val = signature

        # 检查签名参数范围
        if not (1 <= r_val < self.group_order and 1 <= s_val < self.group_order):
            return False

        # 计算消息哈希
        msg_hash = int.from_bytes(hashlib.sha256(message).digest(), 'big')

        # 计算 w = s^(-1) mod n
        w_value = self.compute_modular_inverse(s_val, self.group_order)

        # 计算 u1 = hash * w mod n, u2 = r * w mod n
        u1 = (msg_hash * w_value) % self.group_order
        u2 = (r_val * w_value) % self.group_order

        # 计算验证点
        point1 = self.scalar_point_multiplication(u1, self.base_point)
        point2 = self.scalar_point_multiplication(u2, public_key)
        verification_point = self.perform_point_addition(point1, point2)

        # 验证结果
        return verification_point and verification_point[0] % self.group_order == r_val


class SignatureForgeryAttack:
    """签名伪造攻击器"""

    def __init__(self, target_public_key: Tuple[int, int], curve_order: int):
        self.public_key = target_public_key
        self.curve_order = curve_order

    def execute_forgery_attack(self, original_message: bytes, original_signature: Tuple[int, int],
                               target_message: bytes) -> Tuple[int, int]:
        """
        执行签名伪造攻击

        攻击原理：
        1. 利用固定随机数k的漏洞
        2. 通过已知的(r, s)和消息哈希差推导新的s值
        3. 生成对目标消息的有效签名
        """
        r_val, s_val = original_signature

        # 计算原始消息和目标消息的哈希值
        original_hash = int.from_bytes(hashlib.sha256(original_message).digest(), 'big')
        target_hash = int.from_bytes(hashlib.sha256(target_message).digest(), 'big')

        # 计算哈希差
        hash_difference = (target_hash - original_hash) % self.curve_order

        # 利用固定k漏洞推导新的s值
        # 公式：s_new = s_old + (hash_diff * r^(-1)) mod n
        r_inverse = pow(r_val, -1, self.curve_order)
        s_new = (s_val + hash_difference * r_inverse) % self.curve_order

        return (r_val, s_new)


def demonstrate_attack():
    """演示签名伪造攻击"""
    print("=== 数字签名伪造攻击演示 ===\n")

    # 初始化脆弱的签名系统
    weak_system = WeakSignatureSystem()
    private_key, public_key = weak_system.create_key_pair()

    print(f"私钥: {private_key}")
    print(f"公钥: ({public_key[0]}, {public_key[1]})")

    # 原始消息
    original_message = b"Alice receives 10 BTC from Bob"
    print(f"\n原始消息: {original_message.decode('utf-8')}")

    # 使用固定随机数生成签名（安全漏洞）
    FIXED_NONCE = 987654321
    original_signature = weak_system.generate_signature(original_message, private_key, FIXED_NONCE)
    print(f"原始签名: r={original_signature[0]}")
    print(f"         s={original_signature[1]}")

    # 验证原始签名
    is_valid = weak_system.validate_signature(original_message, original_signature, public_key)
    print(f"原始签名验证: {'✓ 成功' if is_valid else '✗ 失败'}")

    # 目标伪造消息
    forged_message = b"Alice receives 1000 BTC from Bob"
    print(f"\n伪造消息: {forged_message.decode('utf-8')}")

    # 创建攻击器
    attacker = SignatureForgeryAttack(public_key, CURVE_ORDER)

    # 执行伪造攻击
    forged_signature = attacker.execute_forgery_attack(original_message, original_signature, forged_message)
    print(f"伪造签名: r={forged_signature[0]}")
    print(f"         s={forged_signature[1]}")

    # 验证伪造签名
    forged_valid = weak_system.validate_signature(forged_message, forged_signature, public_key)
    print(f"伪造签名验证: {'✓ 成功' if forged_valid else '✗ 失败'}")



if __name__ == "__main__":
    demonstrate_attack()
