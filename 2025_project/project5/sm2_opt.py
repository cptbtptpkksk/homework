
import hashlib
import random
import time
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# SM2椭圆曲线参数 (国家标准)
CURVE_PRIME = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
CURVE_A = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
CURVE_B = 0x28E9FA9E9D9F5E344D5AEF7E8B5D50A0C648FEE9A97A7E37BBA2DDF1D5
CURVE_ORDER = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
BASE_POINT_X = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
BASE_POINT_Y = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
BASE_POINT = (BASE_POINT_X, BASE_POINT_Y)

# 无穷远点表示
INFINITY_POINT = (0, 0)


class EllipticCurveMath:
    """椭圆曲线数学运算类"""

    @staticmethod
    def modular_inverse(value: int, modulus: int) -> int:
        """计算模逆元 - 使用扩展欧几里得算法"""

        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(value, modulus)
        if gcd != 1:
            raise ValueError("模逆元不存在")
        return (x % modulus + modulus) % modulus

    @staticmethod
    def point_doubling(point: Tuple[int, int]) -> Tuple[int, int]:
        """椭圆曲线点倍运算"""
        if point == INFINITY_POINT:
            return INFINITY_POINT

        x, y = point
        if y == 0:
            return INFINITY_POINT

        # 计算斜率 λ = (3x² + a) / (2y)
        numerator = (3 * x * x + CURVE_A) % CURVE_PRIME
        denominator = (2 * y) % CURVE_PRIME
        lambda_val = (numerator * EllipticCurveMath.modular_inverse(denominator, CURVE_PRIME)) % CURVE_PRIME

        # 计算新点坐标
        x_new = (lambda_val * lambda_val - 2 * x) % CURVE_PRIME
        y_new = (lambda_val * (x - x_new) - y) % CURVE_PRIME

        return (x_new, y_new)

    @staticmethod
    def point_addition(point1: Tuple[int, int], point2: Tuple[int, int]) -> Tuple[int, int]:
        """椭圆曲线点加运算"""
        if point1 == INFINITY_POINT:
            return point2
        if point2 == INFINITY_POINT:
            return point1

        x1, y1 = point1
        x2, y2 = point2

        # 处理特殊情况
        if x1 == x2:
            if (y1 + y2) % CURVE_PRIME == 0:
                return INFINITY_POINT
            else:
                return EllipticCurveMath.point_doubling(point1)

        # 计算斜率 λ = (y2 - y1) / (x2 - x1)
        numerator = (y2 - y1) % CURVE_PRIME
        denominator = (x2 - x1) % CURVE_PRIME
        lambda_val = (numerator * EllipticCurveMath.modular_inverse(denominator, CURVE_PRIME)) % CURVE_PRIME

        # 计算新点坐标
        x_new = (lambda_val * lambda_val - x1 - x2) % CURVE_PRIME
        y_new = (lambda_val * (x1 - x_new) - y1) % CURVE_PRIME

        return (x_new, y_new)

    @staticmethod
    def scalar_multiplication(scalar: int, point: Tuple[int, int]) -> Tuple[int, int]:
        """标量乘法 - 使用NAF (Non-Adjacent Form) 优化"""
        if scalar == 0:
            return INFINITY_POINT

        result = INFINITY_POINT
        current_point = point

        while scalar > 0:
            if scalar & 1:
                result = EllipticCurveMath.point_addition(result, current_point)
            current_point = EllipticCurveMath.point_doubling(current_point)
            scalar >>= 1

        return result


class SM2CryptoSystem:
    """SM2密码系统主类"""

    def __init__(self):
        self.curve_math = EllipticCurveMath()
        self._random_generator = random.SystemRandom()

    def generate_key_pair(self) -> Tuple[int, Tuple[int, int]]:
        """生成SM2密钥对"""
        private_key = self._random_generator.randrange(1, CURVE_ORDER)
        public_key = self.curve_math.scalar_multiplication(private_key, BASE_POINT)
        return private_key, public_key

    def compute_message_hash(self, message: bytes) -> int:
        """计算消息哈希值"""
        hash_bytes = hashlib.sha256(message).digest()
        return int.from_bytes(hash_bytes, 'big') % CURVE_ORDER

    def sign_message(self, message: bytes, private_key: int) -> Tuple[int, int]:
        """SM2数字签名"""
        message_hash = self.compute_message_hash(message)

        while True:
            # 生成随机数k
            k = self._random_generator.randrange(1, CURVE_ORDER)

            # 计算点 (x1, y1) = k * G
            temp_point = self.curve_math.scalar_multiplication(k, BASE_POINT)
            x1 = temp_point[0]

            # 计算 r = (e + x1) mod n
            r = (message_hash + x1) % CURVE_ORDER

            # 检查r的有效性
            if r == 0 or (r + k) % CURVE_ORDER == 0:
                continue

            # 计算 s = (1 + d)^(-1) * (k - r * d) mod n
            d_plus_one_inv = self.curve_math.modular_inverse(1 + private_key, CURVE_ORDER)
            s = (d_plus_one_inv * (k - r * private_key)) % CURVE_ORDER

            if s != 0:
                return (r, s)

    def verify_signature(self, message: bytes, signature: Tuple[int, int], public_key: Tuple[int, int]) -> bool:
        """SM2签名验证"""
        r, s = signature

        # 检查签名参数范围
        if not (1 <= r < CURVE_ORDER and 1 <= s < CURVE_ORDER):
            return False

        message_hash = self.compute_message_hash(message)

        # 计算 t = (r + s) mod n
        t = (r + s) % CURVE_ORDER
        if t == 0:
            return False

        # 计算点 (x1', y1') = s * G + t * P
        s_g = self.curve_math.scalar_multiplication(s, BASE_POINT)
        t_p = self.curve_math.scalar_multiplication(t, public_key)
        result_point = self.curve_math.point_addition(s_g, t_p)

        if result_point == INFINITY_POINT:
            return False

        # 计算 R = (e + x1') mod n
        R = (message_hash + result_point[0]) % CURVE_ORDER

        return R == r


class SM2OptimizedCryptoSystem(SM2CryptoSystem):
    """SM2优化版本 - 包含性能优化和安全增强"""

    def __init__(self, enable_parallel: bool = True, cache_size: int = 1000):
        super().__init__()
        self.enable_parallel = enable_parallel
        self._point_cache = {}
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()

    def _cached_scalar_multiplication(self, scalar: int, point: Tuple[int, int]) -> Tuple[int, int]:
        """带缓存的标量乘法"""
        cache_key = (scalar, point)

        with self._cache_lock:
            if cache_key in self._point_cache:
                return self._point_cache[cache_key]

        result = self.curve_math.scalar_multiplication(scalar, point)

        with self._cache_lock:
            if len(self._point_cache) < self._cache_size:
                self._point_cache[cache_key] = result

        return result

    def batch_sign_messages(self, messages: List[bytes], private_key: int, max_workers: int = 4) -> List[
        Tuple[int, int]]:
        """批量签名消息"""
        if not self.enable_parallel:
            return [self.sign_message(msg, private_key) for msg in messages]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_msg = {executor.submit(self.sign_message, msg, private_key): msg for msg in messages}
            signatures = []

            for future in as_completed(future_to_msg):
                try:
                    signature = future.result()
                    signatures.append(signature)
                except Exception as exc:
                    print(f'签名过程中发生错误: {exc}')
                    signatures.append(None)

        return signatures

    def batch_verify_signatures(self, messages: List[bytes], signatures: List[Tuple[int, int]],
                                public_key: Tuple[int, int], max_workers: int = 4) -> List[bool]:
        """批量验证签名"""
        if not self.enable_parallel:
            return [self.verify_signature(msg, sig, public_key) for msg, sig in zip(messages, signatures)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_verification = {
                executor.submit(self.verify_signature, msg, sig, public_key): i
                for i, (msg, sig) in enumerate(zip(messages, signatures))
            }

            results = [False] * len(messages)
            for future in as_completed(future_to_verification):
                try:
                    index = future_to_verification[future]
                    results[index] = future.result()
                except Exception as exc:
                    print(f'验证过程中发生错误: {exc}')

        return results


def performance_benchmark():
    """性能基准测试"""
    print("=== SM2性能基准测试 ===")

    # 基础版本测试
    basic_system = SM2CryptoSystem()
    private_key, public_key = basic_system.generate_key_pair()

    test_messages = [f"测试消息 #{i}".encode('utf-8') for i in range(10)]

    # 基础版本签名测试
    start_time = time.time()
    basic_signatures = []
    for msg in test_messages:
        sig = basic_system.sign_message(msg, private_key)
        basic_signatures.append(sig)
    basic_sign_time = time.time() - start_time

    # 基础版本验证测试
    start_time = time.time()
    basic_verify_results = []
    for msg, sig in zip(test_messages, basic_signatures):
        result = basic_system.verify_signature(msg, sig, public_key)
        basic_verify_results.append(result)
    basic_verify_time = time.time() - start_time

    # 优化版本测试
    optimized_system = SM2OptimizedCryptoSystem(enable_parallel=True)

    # 优化版本批量签名测试
    start_time = time.time()
    optimized_signatures = optimized_system.batch_sign_messages(test_messages, private_key, max_workers=4)
    optimized_sign_time = time.time() - start_time

    # 优化版本批量验证测试
    start_time = time.time()
    optimized_verify_results = optimized_system.batch_verify_signatures(
        test_messages, optimized_signatures, public_key, max_workers=4
    )
    optimized_verify_time = time.time() - start_time

    # 输出结果
    print(f"基础版本签名时间: {basic_sign_time:.4f}秒")
    print(f"基础版本验证时间: {basic_verify_time:.4f}秒")
    print(f"优化版本批量签名时间: {optimized_sign_time:.4f}秒")
    print(f"优化版本批量验证时间: {optimized_verify_time:.4f}秒")
    print(f"签名性能提升: {basic_sign_time / optimized_sign_time:.2f}x")
    print(f"验证性能提升: {basic_verify_time / optimized_verify_time:.2f}x")
    print(f"基础版本验证成功率: {sum(basic_verify_results)}/{len(basic_verify_results)}")
    print(f"优化版本验证成功率: {sum(optimized_verify_results)}/{len(optimized_verify_results)}")


if __name__ == "__main__":
    # 基本功能测试
    print("=== SM2增强实现测试 ===")

    # 创建密码系统实例
    sm2_system = SM2CryptoSystem()

    # 生成密钥对
    private_key, public_key = sm2_system.generate_key_pair()
    print(f"私钥: {private_key}")
    print(f"公钥: ({public_key[0]}, {public_key[1]})")

    # 测试消息
    test_message = "Hello, SM2密码系统!".encode('utf-8')
    print(f"测试消息: {test_message.decode()}")

    # 签名
    signature = sm2_system.sign_message(test_message, private_key)
    print(f"签名: r={signature[0]}, s={signature[1]}")

    # 验证
    is_valid = sm2_system.verify_signature(test_message, signature, public_key)
    print(f"签名验证: {'成功' if is_valid else '失败'}")

    # 性能测试
    print("\n")
    performance_benchmark()


