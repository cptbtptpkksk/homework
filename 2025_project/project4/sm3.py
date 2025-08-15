
import struct
from typing import List, Tuple


class SM3Hash:
    """SM3哈希算法实现类"""

    # 初始向量常量
    INITIAL_VECTOR = [
        0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
        0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
    ]

    # 轮常量
    ROUND_CONSTANTS = [
        0x79CC4519 if i < 16 else 0x7A879D8A for i in range(64)
    ]

    def __init__(self):
        """初始化SM3哈希对象"""
        self.state = self.INITIAL_VECTOR.copy()

    @staticmethod
    def circular_left_shift(value: int, shift: int) -> int:
        """32位循环左移操作"""
        return ((value << shift) | (value >> (32 - shift))) & 0xFFFFFFFF

    @staticmethod
    def boolean_function_f(x: int, y: int, z: int, round_num: int) -> int:
        """布尔函数F"""
        if round_num < 16:
            return x ^ y ^ z
        else:
            return (x & y) | (x & z) | (y & z)

    @staticmethod
    def boolean_function_g(x: int, y: int, z: int, round_num: int) -> int:
        """布尔函数G"""
        if round_num < 16:
            return x ^ y ^ z
        else:
            return (x & y) | (~x & z)

    @staticmethod
    def permutation_p0(x: int) -> int:
        """置换函数P0"""
        return x ^ SM3Hash.circular_left_shift(x, 9) ^ SM3Hash.circular_left_shift(x, 17)

    @staticmethod
    def permutation_p1(x: int) -> int:
        """置换函数P1"""
        return x ^ SM3Hash.circular_left_shift(x, 15) ^ SM3Hash.circular_left_shift(x, 23)

    def message_padding(self, message: bytes) -> bytes:
        """消息填充函数"""
        original_length = len(message)
        bit_length = original_length * 8

        # 添加填充位
        padded_message = bytearray(message)
        padded_message.append(0x80)  # 添加1

        # 添加0直到长度满足要求
        while (len(padded_message) % 64) != 56:
            padded_message.append(0x00)

        # 添加原始消息长度（64位）
        padded_message.extend(struct.pack('>Q', bit_length))

        return bytes(padded_message)

    def message_expansion(self, block: bytes) -> Tuple[List[int], List[int]]:
        """消息扩展函数"""
        # 将512位块分解为16个32位字
        words = []
        for i in range(16):
            word = struct.unpack('>I', block[i * 4:(i + 1) * 4])[0]
            words.append(word)

        # 扩展生成68个字
        for j in range(16, 68):
            temp = (words[j - 16] ^ words[j - 9] ^
                    self.circular_left_shift(words[j - 3], 15))
            word = (self.permutation_p1(temp) ^
                    self.circular_left_shift(words[j - 13], 7) ^
                    words[j - 6])
            words.append(word & 0xFFFFFFFF)

        # 生成64个W'字
        words_prime = []
        for j in range(64):
            word_prime = words[j] ^ words[j + 4]
            words_prime.append(word_prime)

        return words, words_prime

    def compression_function(self, block: bytes) -> None:
        """压缩函数"""
        # 消息扩展
        words, words_prime = self.message_expansion(block)

        # 初始化工作变量
        a, b, c, d, e, f, g, h = self.state

        # 64轮迭代
        for j in range(64):
            # 计算中间变量
            ss1 = self.circular_left_shift(
                (self.circular_left_shift(a, 12) + e +
                 self.circular_left_shift(self.ROUND_CONSTANTS[j], j % 32)) & 0xFFFFFFFF,
                7
            )
            ss2 = ss1 ^ self.circular_left_shift(a, 12)

            tt1 = (self.boolean_function_f(a, b, c, j) + d + ss2 + words_prime[j]) & 0xFFFFFFFF
            tt2 = (self.boolean_function_g(e, f, g, j) + h + ss1 + words[j]) & 0xFFFFFFFF

            # 更新工作变量
            d = c
            c = self.circular_left_shift(b, 9)
            b = a
            a = tt1

            h = g
            g = self.circular_left_shift(f, 19)
            f = e
            e = self.permutation_p0(tt2)

        self.state[0] ^= a
        self.state[1] ^= b
        self.state[2] ^= c
        self.state[3] ^= d
        self.state[4] ^= e
        self.state[5] ^= f
        self.state[6] ^= g
        self.state[7] ^= h

    def compute_hash(self, message: bytes) -> str:
        """计算SM3哈希值"""
        # 重置状态
        self.state = self.INITIAL_VECTOR.copy()

        # 消息填充
        padded_message = self.message_padding(message)

        # 分块处理
        block_size = 64
        for i in range(0, len(padded_message), block_size):
            block = padded_message[i:i + block_size]
            self.compression_function(block)

        # 返回十六进制字符串
        return ''.join(f'{x:08x}' for x in self.state)


def sm3_hash(message: bytes) -> str:
    """SM3哈希函数接口"""
    hasher = SM3Hash()
    return hasher.compute_hash(message)


if __name__ == '__main__':
    # 测试用例
    test_cases = [
        b"abc",
        b"",
        b"hello world",
        b"SM3test"
    ]

    print("SM3哈希算法测试结果：")
    print("=" * 50)

    for i, test_msg in enumerate(test_cases, 1):
        hash_result = sm3_hash(test_msg)
        print(f"测试 {i}:")
        print(f"  消息: {test_msg}")
        print(f"  哈希值: {hash_result}")
        print()
