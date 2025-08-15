#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <array>

using namespace std;

// SM3算法实现
class SM3LengthExtensionAttack {
private:
    // 初始向量
    static constexpr uint32_t IV[8] = {
        0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
        0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
    };

    // 轮常量
    static constexpr uint32_t T[64] = {
        0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
        0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
        0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
        0x79CC4519, 0x79CC4519, 0x79CC4519, 0x79CC4519,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A,
        0x7A879D8A, 0x7A879D8A, 0x7A879D8A, 0x7A879D8A
    };

    // 工具函数
    inline static uint32_t rotate_left(uint32_t x, int n) {
        return (x << n) | (x >> (32 - n));
    }

    inline static uint32_t bool_func_f(uint32_t x, uint32_t y, uint32_t z, int round) {
        return (round < 16) ? (x ^ y ^ z) : ((x & y) | (x & z) | (y & z));
    }

    inline static uint32_t bool_func_g(uint32_t x, uint32_t y, uint32_t z, int round) {
        return (round < 16) ? (x ^ y ^ z) : ((x & y) | (~x & z));
    }

    inline static uint32_t perm_p0(uint32_t x) {
        return x ^ rotate_left(x, 9) ^ rotate_left(x, 17);
    }

    inline static uint32_t perm_p1(uint32_t x) {
        return x ^ rotate_left(x, 15) ^ rotate_left(x, 23);
    }

    // 消息扩展
    static void message_expansion(const uint8_t* block, uint32_t* words, uint32_t* words_prime) {
        // 初始16个字
        for (int i = 0; i < 16; ++i) {
            words[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
                (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
                (static_cast<uint32_t>(block[4 * i + 3]));
        }

        // 扩展到68个字
        for (int j = 16; j < 68; ++j) {
            uint32_t temp = words[j - 16] ^ words[j - 9] ^ rotate_left(words[j - 3], 15);
            words[j] = perm_p1(temp) ^ rotate_left(words[j - 13], 7) ^ words[j - 6];
        }

        // 生成64个W'字
        for (int j = 0; j < 64; ++j) {
            words_prime[j] = words[j] ^ words[j + 4];
        }
    }

    // 压缩函数
    static void compression_function(array<uint32_t, 8>& state, const uint8_t* block) {
        uint32_t words[68], words_prime[64];
        message_expansion(block, words, words_prime);

        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

        // 64轮迭代
        for (int j = 0; j < 64; ++j) {
            uint32_t ss1 = rotate_left(
                (rotate_left(a, 12) + e + rotate_left(T[j], j % 32)) & 0xFFFFFFFF,
                7
            );
            uint32_t ss2 = ss1 ^ rotate_left(a, 12);

            uint32_t tt1 = (bool_func_f(a, b, c, j) + d + ss2 + words_prime[j]) & 0xFFFFFFFF;
            uint32_t tt2 = (bool_func_g(e, f, g, j) + h + ss1 + words[j]) & 0xFFFFFFFF;

            d = c;
            c = rotate_left(b, 9);
            b = a;
            a = tt1;

            h = g;
            g = rotate_left(f, 19);
            f = e;
            e = perm_p0(tt2);
        }

        // 状态更新
        state[0] ^= a; state[1] ^= b; state[2] ^= c; state[3] ^= d;
        state[4] ^= e; state[5] ^= f; state[6] ^= g; state[7] ^= h;
    }

public:
    // 消息填充
    static vector<uint8_t> create_padding(const string& message, uint64_t total_bit_length) {
        vector<uint8_t> padded(message.begin(), message.end());

        // 添加填充位
        padded.push_back(0x80);
        while ((padded.size() % 64) != 56) {
            padded.push_back(0x00);
        }

        // 添加总长度
        for (int i = 7; i >= 0; --i) {
            padded.push_back(static_cast<uint8_t>(total_bit_length >> (i * 8)));
        }

        return padded;
    }

    // 标准SM3哈希计算
    static string compute_hash(const string& message) {
        array<uint32_t, 8> state;
        memcpy(state.data(), IV, sizeof(IV));

        auto padded = create_padding(message, message.length() * 8);

        for (size_t i = 0; i < padded.size(); i += 64) {
            compression_function(state, &padded[i]);
        }

        return state_to_hex(state);
    }

    // 从哈希值恢复状态
    static array<uint32_t, 8> hash_to_state(const string& hash) {
        array<uint32_t, 8> state;
        for (int i = 0; i < 8; ++i) {
            string hex_part = hash.substr(i * 8, 8);
            state[i] = stoul(hex_part, nullptr, 16);
        }
        return state;
    }

    // 继续哈希计算（用于长度扩展攻击）
    static string continue_hash(const array<uint32_t, 8>& initial_state,
        const string& additional_data,
        uint64_t total_bit_length) {
        array<uint32_t, 8> state = initial_state;

        auto padded = create_padding(additional_data, total_bit_length);

        for (size_t i = 0; i < padded.size(); i += 64) {
            compression_function(state, &padded[i]);
        }

        return state_to_hex(state);
    }

    // 状态转换为十六进制字符串
    static string state_to_hex(const array<uint32_t, 8>& state) {
        ostringstream oss;
        for (int i = 0; i < 8; ++i) {
            oss << hex << setw(8) << setfill('0') << state[i];
        }
        return oss.str();
    }

    // 执行长度扩展攻击
    static void perform_attack(const string& original_message,
        const string& additional_data,
        const string& original_hash) {
        cout << "=== SM3长度扩展攻击演示 ===" << endl;
        cout << "原始消息: " << original_message << endl;
        cout << "原始哈希: " << original_hash << endl;
        cout << "附加数据: " << additional_data << endl;
        cout << endl;

        // 步骤1: 从哈希值恢复状态
        auto recovered_state = hash_to_state(original_hash);
        cout << "步骤1: 从哈希值恢复内部状态" << endl;
        cout << "恢复的状态: " << state_to_hex(recovered_state) << endl;
        cout << endl;

        // 步骤2: 计算原始消息的填充
        auto original_padding = create_padding(original_message, original_message.length() * 8);
        size_t padding_size = original_padding.size() - original_message.length();

        cout << "步骤2: 计算原始消息填充" << endl;
        cout << "原始消息长度: " << original_message.length() << " 字节" << endl;
        cout << "填充大小: " << padding_size << " 字节" << endl;
        cout << endl;

        // 步骤3: 计算伪造消息的总长度
        uint64_t forged_total_length = (original_message.length() + padding_size + additional_data.length()) * 8;

        cout << "步骤3: 计算伪造消息总长度" << endl;
        cout << "伪造消息总长度: " << forged_total_length << " 位" << endl;
        cout << endl;

        // 步骤4: 使用恢复的状态继续哈希计算
        string forged_hash = continue_hash(recovered_state, additional_data, forged_total_length);

        cout << "步骤4: 执行长度扩展攻击" << endl;
        cout << "攻击者伪造的哈希: " << forged_hash << endl;
        cout << endl;

        // 步骤5: 验证攻击结果
        string real_extended_message = original_message +
            string(reinterpret_cast<const char*>(original_padding.data() + original_message.length()),
                padding_size) +
            additional_data;
        string real_hash = compute_hash(real_extended_message);

        cout << "步骤5: 验证攻击结果" << endl;
        cout << "真实扩展消息哈希: " << real_hash << endl;
        cout << endl;

        // 结果对比
        bool attack_success = (forged_hash == real_hash);
        cout << "=== 攻击结果 ===" << endl;
        cout << "攻击" << (attack_success ? "成功" : "失败") << "!" << endl;
        cout << "哈希值匹配: " << (attack_success ? "是" : "否") << endl;
        cout << endl;

        
    }
};

int main() {
    // 测试用例
    string original_msg = "secret_key";
    string extra_data = "&admin=true";

    // 计算原始哈希
    string original_hash = SM3LengthExtensionAttack::compute_hash(original_msg);

    // 执行长度扩展攻击
    SM3LengthExtensionAttack::perform_attack(original_msg, extra_data, original_hash);

    // 额外测试用例
    cout << "\n=== 额外测试用例 ===" << endl;

    vector<pair<string, string>> test_cases = {
        {"abc", "def"},
        {"hello", "world"},
        {"test", "data"}
    };

    for (const auto& test_case : test_cases) {
        string original = test_case.first;
        string extra = test_case.second;
        string hash = SM3LengthExtensionAttack::compute_hash(original);

        cout << "测试: " << original << " + " << extra << endl;
        SM3LengthExtensionAttack::perform_attack(original, extra, hash);
        cout << "---" << endl;
    }

    return 0;
}
