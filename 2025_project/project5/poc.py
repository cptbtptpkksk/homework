import hashlib
import random
from typing import Optional, Tuple, List

# ---------- SM2 椭圆曲线参数 ----------
prime_p = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF", 16)
curve_a = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC", 16)
curve_b = int("28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93", 16)
base_Gx = int("32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7", 16)
base_Gy = int("BC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0", 16)
order_n = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123", 16)


# ---------- 模逆计算 ----------
def modular_inverse(num: int, mod: int) -> int:
    """计算模逆：返回x使得(num * x) ≡ 1 mod mod"""
    num %= mod
    if num == 0:
        raise ZeroDivisionError("无法计算0的模逆")
    return pow(num, -1, mod)


# ---------- 椭圆曲线点运算（仿射坐标） ----------
def ec_point_add(point_p: Optional[Tuple[int, int]], point_q: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """椭圆曲线点加法：返回P + Q的结果，无穷远点用None表示"""
    if point_p is None:
        return point_q
    if point_q is None:
        return point_p

    x1, y1 = point_p
    x2, y2 = point_q

    # 处理相反点情况（P + (-P) = 无穷远点）
    if x1 == x2 and (y1 + y2) % prime_p == 0:
        return None

    # 计算斜率λ
    if point_p != point_q:
        lambda_val = ((y2 - y1) * modular_inverse(x2 - x1, prime_p)) % prime_p
    else:
        lambda_val = ((3 * x1 * x1 + curve_a) * modular_inverse(2 * y1, prime_p)) % prime_p

    # 计算相加结果
    x3 = (lambda_val * lambda_val - x1 - x2) % prime_p
    y3 = (lambda_val * (x1 - x3) - y1) % prime_p
    return (x3, y3)


# ---------- w-NAF 优化相关函数 ----------
def compute_wnaf(scalar: int, window: int = 5) -> List[int]:
    """计算标量的w-NAF表示，用于高效点乘"""
    if scalar == 0:
        return [0]
    digits = []
    while scalar > 0:
        if scalar & 1:
            mod_result = scalar % (1 << window)
            if mod_result & (1 << (window - 1)):
                digit = mod_result - (1 << window)
            else:
                digit = mod_result
            digits.append(digit)
            scalar -= digit
        else:
            digits.append(0)
        scalar >>= 1
    return digits


def precompute_naf_base(base_point: Tuple[int, int], window: int = 5) -> List[Tuple[int, int]]:
    """预计算w-NAF所需的基点倍数，加速点乘运算"""
    max_odd = (1 << window) - 1
    precomputed = []
    current = base_point
    precomputed.append(current)
    two_base = ec_point_add(base_point, base_point)  # 2G

    odd = 3
    while odd <= max_odd:
        current = ec_point_add(current, two_base)  # 累加2G得到3G,5G,...
        precomputed.append(current)
        odd += 2
    return precomputed


def scalar_multiply_wnaf(scalar: int, base_point: Tuple[int, int],
                         window: int = 5, precomputed: Optional[List[Tuple[int, int]]] = None) -> Optional[
    Tuple[int, int]]:
    """使用w-NAF方法高效计算标量乘法k*P"""
    if scalar % order_n == 0:
        return None  # 无穷远点
    if precomputed is None:
        precomputed = precompute_naf_base(base_point, window)

    wnaf_digits = compute_wnaf(scalar, window)
    result = None

    for digit in reversed(wnaf_digits):
        result = ec_point_add(result, result)  # 双倍点
        if digit != 0:
            if digit > 0:
                idx = (digit - 1) // 2
                result = ec_point_add(result, precomputed[idx])
            else:
                idx = ((-digit) - 1) // 2
                # 取点的逆（y坐标取负）
                neg_point = (precomputed[idx][0], (-precomputed[idx][1]) % prime_p)
                result = ec_point_add(result, neg_point)
    return result


# ---------- 哈希函数封装 ----------
def message_hash(data: bytes) -> bytes:
    """消息哈希函数，使用SHA-256"""
    return hashlib.sha256(data).digest()


# ---------- SM2密钥对类（优化实现） ----------
class SM2KeyPair:
    """SM2密钥对类，支持签名生成与验证"""

    def __init__(self, private_key: Optional[int] = None, window: int = 5):
        """初始化密钥对，未指定私钥则随机生成"""
        if private_key is None:
            self.private_key = random.randrange(1, order_n)
        else:
            self.private_key = private_key % order_n
        self.window_size = window
        # 预计算基点G的倍数，加速后续运算
        self.precomputed_G = precompute_naf_base((base_Gx, base_Gy), window)
        # 计算公钥：public_key = private_key * G
        self.public_key = scalar_multiply_wnaf(
            self.private_key, (base_Gx, base_Gy), window, self.precomputed_G
        )

    def sign_using_nonce(self, zm_hash: bytes, nonce_k: int) -> Tuple[int, int]:
        """使用指定随机数k生成签名"""
        # 计算e = H(Z || M) mod n
        e_hash = int.from_bytes(message_hash(zm_hash), 'big') % order_n
        # 计算k*G
        kG = scalar_multiply_wnaf(nonce_k, (base_Gx, base_Gy), self.window_size, self.precomputed_G)
        if kG is None:
            raise ValueError("随机数k导致k*G为无穷远点，无效")
        x1 = kG[0] % order_n
        # 计算r = (e + x1) mod n
        r = (e_hash + x1) % order_n
        # 计算s = (1/(1+d)) * (k - r*d) mod n
        s_numerator = (nonce_k - r * self.private_key) % order_n
        s_denominator = modular_inverse(1 + self.private_key, order_n)
        s = (s_numerator * s_denominator) % order_n
        return (r, s)

    def generate_signature(self, zm_hash: bytes) -> Tuple[int, int, int]:
        """生成签名（返回r, s和使用的随机数k）"""
        nonce_k = random.randrange(1, order_n)
        r, s = self.sign_using_nonce(zm_hash, nonce_k)
        return (r, s, nonce_k)

    def verify_signature(self, zm_hash: bytes, signature: Tuple[int, int]) -> bool:
        """验证签名有效性"""
        r, s = signature
        # 检查r和s的范围
        if not (1 <= r <= order_n - 1 and 1 <= s <= order_n - 1):
            return False
        # 计算e = H(Z || M) mod n
        e_hash = int.from_bytes(message_hash(zm_hash), 'big') % order_n
        t = (r + s) % order_n
        if t == 0:
            return False
        # 计算s*G + t*P
        sG = scalar_multiply_wnaf(s, (base_Gx, base_Gy), self.window_size, self.precomputed_G)
        tP = scalar_multiply_wnaf(t, self.public_key, self.window_size)
        sum_point = ec_point_add(sG, tP)
        if sum_point is None:
            return False
        x2 = sum_point[0] % order_n
        # 验证r是否等于(e + x2) mod n
        return r == (e_hash + x2) % order_n


# ---------- 私钥恢复函数 ----------
def recover_private_key_from_reused_nonce(r1: int, s1: int, r2: int, s2: int) -> Optional[int]:
    """从重复使用的随机数攻击中恢复私钥d"""
    # 推导公式：d = (s1 - s2) / (s2 + r2 - s1 - r1) mod n
    numerator = (s1 - s2) % order_n
    denominator = (s2 + r2 - s1 - r1) % order_n
    if denominator % order_n == 0:
        return None  # 分母为0，无法计算
    return (numerator * modular_inverse(denominator, order_n)) % order_n


# ---------- 演示案例 ----------
def sm2_nonce_reuse_demo():
    print("=== SM2随机数重用漏洞演示 ===")
    # 生成用户密钥对
    user_key = SM2KeyPair()
    print(f"用户公钥P.x = {hex(user_key.public_key[0])}")

    # 两条测试消息
    msg1 = b"vciuhf"
    msg2 = b"dajhkf"

    # 模拟攻击者观察到同一随机数被重复使用
    reused_nonce = random.randrange(1, order_n)
    sig1 = user_key.sign_using_nonce(msg1, reused_nonce)
    sig2 = user_key.sign_using_nonce(msg2, reused_nonce)
    r1, s1 = sig1
    r2, s2 = sig2
    print(f"签名1 (r,s): {r1}, {s1}")
    print(f"签名2 (r,s): {r2}, {s2}")

    # 攻击者恢复私钥
    recovered_d = recover_private_key_from_reused_nonce(r1, s1, r2, s2)
    if recovered_d is None:
        print("私钥恢复失败（特殊情况：分母为0）")
        return

    print(f"恢复的私钥d: {hex(recovered_d)}")
    print(f"实际用户私钥d: {hex(user_key.private_key)}")
    print(f"恢复是否成功?: {recovered_d == user_key.private_key}")

    # 使用恢复的私钥伪造新签名
    attacker_key = SM2KeyPair(private_key=recovered_d)
    forged_r, forged_s = attacker_key.sign_using_nonce(b"vciuhf", random.randrange(1, order_n))
    print(f"伪造的签名 (r,s): {forged_r}, {forged_s}")

    # 验证伪造签名是否通过
    verify_result = user_key.verify_signature(b"dajhkf", (forged_r, forged_s))
    print(f"伪造签名验证结果（应返回True）: {verify_result}")


if __name__ == "__main__":
    sm2_nonce_reuse_demo()