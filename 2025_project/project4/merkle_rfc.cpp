#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <array>

using namespace std;

// ====================== SM3哈希算法实现 ======================
// 初始向量定义
static const uint32_t SM3_IV[8] = {
    0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
    0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
};

// 常量T
static const uint32_t SM3_T[64] = {
    0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,
    0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A
};

// 左旋转操作
inline uint32_t left_rotate(uint32_t value, int shift) {
    return (value << shift) | (value >> (32 - shift));
}

// 置换函数P0
inline uint32_t permute_p0(uint32_t value) {
    return value ^ left_rotate(value, 9) ^ left_rotate(value, 17);
}

// 置换函数P1
inline uint32_t permute_p1(uint32_t value) {
    return value ^ left_rotate(value, 15) ^ left_rotate(value, 23);
}

// 布尔函数FF
inline uint32_t bool_ff(uint32_t x, uint32_t y, uint32_t z, int round) {
    if (round < 16) {
        return x ^ y ^ z;
    }
    else {
        return (x & y) | (x & z) | (y & z);
    }
}

// 布尔函数GG
inline uint32_t bool_gg(uint32_t x, uint32_t y, uint32_t z, int round) {
    if (round < 16) {
        return x ^ y ^ z;
    }
    else {
        return (x & y) | ((~x) & z);
    }
}

// 将哈希结果转换为十六进制字符串
std::string hash_to_hex(const std::array<uint32_t, 8>& hash_values) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < 8; ++i) {
        oss << std::setw(8) << hash_values[i];
    }
    return oss.str();
}

// 压缩函数
void sm3_compress(array<uint32_t, 8>& hash_state, const uint8_t block[64]) {
    uint32_t W[68], W_[64];

    // 消息扩展
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)block[4 * i] << 24) |
            ((uint32_t)block[4 * i + 1] << 16) |
            ((uint32_t)block[4 * i + 2] << 8) |
            ((uint32_t)block[4 * i + 3]);
    }

    for (int j = 16; j < 68; ++j) {
        W[j] = permute_p1(W[j - 16] ^ W[j - 9] ^ left_rotate(W[j - 3], 15)) ^
            left_rotate(W[j - 13], 7) ^ W[j - 6];
    }

    for (int j = 0; j < 64; ++j) {
        W_[j] = W[j] ^ W[j + 4];
    }

    // 压缩迭代
    uint32_t A = hash_state[0], B = hash_state[1], C = hash_state[2], D = hash_state[3];
    uint32_t E = hash_state[4], F = hash_state[5], G = hash_state[6], H = hash_state[7];

    for (int j = 0; j < 64; ++j) {
        uint32_t SS1 = left_rotate(((left_rotate(A, 12) + E + left_rotate(SM3_T[j], j % 32)) & 0xFFFFFFFF), 7);
        uint32_t SS2 = SS1 ^ left_rotate(A, 12);
        uint32_t TT1 = (bool_ff(A, B, C, j) + D + SS2 + W_[j]) & 0xFFFFFFFF;
        uint32_t TT2 = (bool_gg(E, F, G, j) + H + SS1 + W[j]) & 0xFFFFFFFF;

        D = C;
        C = left_rotate(B, 9);
        B = A;
        A = TT1;

        H = G;
        G = left_rotate(F, 19);
        F = E;
        E = permute_p0(TT2);
    }

    // 更新哈希状态
    hash_state[0] ^= A;
    hash_state[1] ^= B;
    hash_state[2] ^= C;
    hash_state[3] ^= D;
    hash_state[4] ^= E;
    hash_state[5] ^= F;
    hash_state[6] ^= G;
    hash_state[7] ^= H;
}

// 计算SM3哈希
std::string calculate_sm3(const string& message) {
    array<uint32_t, 8> hash_state;
    memcpy(hash_state.data(), SM3_IV, sizeof(SM3_IV));

    // 消息填充
    uint64_t bit_length = (uint64_t)message.size() * 8;
    vector<uint8_t> data_buffer(message.begin(), message.end());

    data_buffer.push_back(0x80);
    while ((data_buffer.size() % 64) != 56) {
        data_buffer.push_back(0x00);
    }

    // 添加长度信息
    for (int i = 7; i >= 0; --i) {
        data_buffer.push_back((uint8_t)(bit_length >> (i * 8)));
    }

    // 处理所有数据块
    for (size_t i = 0; i < data_buffer.size(); i += 64) {
        sm3_compress(hash_state, &data_buffer[i]);
    }

    return hash_to_hex(hash_state);
}

// ====================== RFC6962 Merkle树实现 ======================
// 计算叶子节点哈希
std::string compute_leaf_hash(const std::string& data) {
    return calculate_sm3(std::string(1, '\x00') + data);
}

// 计算内部节点哈希
std::string compute_node_hash(const string& left, const string& right) {
    return calculate_sm3(string(1, '\x01') + left + right);
}

class MerkleTree {
private:
    vector<vector<string>> tree_layers;  // 存储树的各层节点，tree_layers[0]为叶子节点层

public:
    // 构造函数，从叶子数据构建Merkle树
    MerkleTree(const vector<string>& leaf_nodes) {
        tree_layers.push_back(leaf_nodes);
        construct_tree();
    }

    // 构建Merkle树
    void construct_tree() {
        // 从叶子层开始向上构建，直到根节点
        while (tree_layers.back().size() > 1) {
            const vector<string>& prev_layer = tree_layers.back();
            vector<string> current_layer;

            // 每两个节点合并为一个父节点
            for (size_t i = 0; i < prev_layer.size(); i += 2) {
                if (i + 1 < prev_layer.size()) {
                    current_layer.push_back(compute_node_hash(prev_layer[i], prev_layer[i + 1]));
                }
                else {
                    // 处理奇数个节点的情况，最后一个节点与自身合并
                    current_layer.push_back(compute_node_hash(prev_layer[i], prev_layer[i]));
                }
            }

            tree_layers.push_back(current_layer);
        }
    }

    // 获取根哈希
    std::string get_root_hash() const {
        return tree_layers.back()[0];
    }

    // 生成指定索引叶子节点的存在性证明
    std::vector<string> generate_proof(size_t index) const {
        vector<string> proof;
        size_t current_index = index;

        // 从叶子层向上收集证明节点
        for (size_t layer = 0; layer < tree_layers.size() - 1; ++layer) {
            size_t sibling_index = (current_index % 2 == 0) ? current_index + 1 : current_index - 1;

            if (sibling_index < tree_layers[layer].size()) {
                proof.push_back(tree_layers[layer][sibling_index]);
            }

            current_index /= 2;
        }

        return proof;
    }

    // 验证存在性证明
    bool validate_proof(const string& leaf_hash, const vector<string>& proof,
        size_t index, const string& root_hash) const {
        string current_hash = leaf_hash;
        size_t current_index = index;

        // 逐层计算哈希，验证是否与根哈希一致
        for (size_t i = 0; i < proof.size(); ++i) {
            if (current_index % 2 == 0) {
                current_hash = compute_node_hash(current_hash, proof[i]);
            }
            else {
                current_hash = compute_node_hash(proof[i], current_hash);
            }
            current_index /= 2;
        }

        return current_hash == root_hash;
    }
};

// ====================== 主测试函数 ======================
int main() {
    const size_t LEAF_COUNT = 100000;  // 叶子节点数量
    vector<string> leaf_hashes;
    leaf_hashes.reserve(LEAF_COUNT);

    // 生成叶子节点哈希
    for (size_t i = 0; i < LEAF_COUNT; ++i) {
        leaf_hashes.push_back(compute_leaf_hash("leaf_data_" + to_string(i)));
    }

    // 构建Merkle树并计时
    auto start_time = chrono::high_resolution_clock::now();
    MerkleTree merkle_tree(leaf_hashes);
    auto end_time = chrono::high_resolution_clock::now();

    // 输出构建信息
    double build_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "[Merkle树构建完成] 叶子节点数=" << LEAF_COUNT
        << " 根哈希=" << merkle_tree.get_root_hash().substr(0, 16) << "..." << endl;
    cout << "构建时间=" << build_duration << " 秒" << endl;

    // 测试存在性证明
    size_t test_index = 12345;  // 测试的叶子节点索引
    vector<string> inclusion_proof = merkle_tree.generate_proof(test_index);
    bool is_valid = merkle_tree.validate_proof(leaf_hashes[test_index], inclusion_proof,
        test_index, merkle_tree.get_root_hash());

    cout << "验证叶子节点 " << test_index << " -> " << (is_valid ? "成功" : "失败") << endl;

    return 0;
}
