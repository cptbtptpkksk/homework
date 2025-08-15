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

// ====================== SM3��ϣ�㷨ʵ�� ======================
// ��ʼ��������
static const uint32_t SM3_IV[8] = {
    0x7380166F, 0x4914B2B9, 0x172442D7, 0xDA8A0600,
    0xA96F30BC, 0x163138AA, 0xE38DEE4D, 0xB0FB0E4E
};

// ����T
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

// ����ת����
inline uint32_t left_rotate(uint32_t value, int shift) {
    return (value << shift) | (value >> (32 - shift));
}

// �û�����P0
inline uint32_t permute_p0(uint32_t value) {
    return value ^ left_rotate(value, 9) ^ left_rotate(value, 17);
}

// �û�����P1
inline uint32_t permute_p1(uint32_t value) {
    return value ^ left_rotate(value, 15) ^ left_rotate(value, 23);
}

// ��������FF
inline uint32_t bool_ff(uint32_t x, uint32_t y, uint32_t z, int round) {
    if (round < 16) {
        return x ^ y ^ z;
    }
    else {
        return (x & y) | (x & z) | (y & z);
    }
}

// ��������GG
inline uint32_t bool_gg(uint32_t x, uint32_t y, uint32_t z, int round) {
    if (round < 16) {
        return x ^ y ^ z;
    }
    else {
        return (x & y) | ((~x) & z);
    }
}

// ����ϣ���ת��Ϊʮ�������ַ���
std::string hash_to_hex(const std::array<uint32_t, 8>& hash_values) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < 8; ++i) {
        oss << std::setw(8) << hash_values[i];
    }
    return oss.str();
}

// ѹ������
void sm3_compress(array<uint32_t, 8>& hash_state, const uint8_t block[64]) {
    uint32_t W[68], W_[64];

    // ��Ϣ��չ
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

    // ѹ������
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

    // ���¹�ϣ״̬
    hash_state[0] ^= A;
    hash_state[1] ^= B;
    hash_state[2] ^= C;
    hash_state[3] ^= D;
    hash_state[4] ^= E;
    hash_state[5] ^= F;
    hash_state[6] ^= G;
    hash_state[7] ^= H;
}

// ����SM3��ϣ
std::string calculate_sm3(const string& message) {
    array<uint32_t, 8> hash_state;
    memcpy(hash_state.data(), SM3_IV, sizeof(SM3_IV));

    // ��Ϣ���
    uint64_t bit_length = (uint64_t)message.size() * 8;
    vector<uint8_t> data_buffer(message.begin(), message.end());

    data_buffer.push_back(0x80);
    while ((data_buffer.size() % 64) != 56) {
        data_buffer.push_back(0x00);
    }

    // ��ӳ�����Ϣ
    for (int i = 7; i >= 0; --i) {
        data_buffer.push_back((uint8_t)(bit_length >> (i * 8)));
    }

    // �����������ݿ�
    for (size_t i = 0; i < data_buffer.size(); i += 64) {
        sm3_compress(hash_state, &data_buffer[i]);
    }

    return hash_to_hex(hash_state);
}

// ====================== RFC6962 Merkle��ʵ�� ======================
// ����Ҷ�ӽڵ��ϣ
std::string compute_leaf_hash(const std::string& data) {
    return calculate_sm3(std::string(1, '\x00') + data);
}

// �����ڲ��ڵ��ϣ
std::string compute_node_hash(const string& left, const string& right) {
    return calculate_sm3(string(1, '\x01') + left + right);
}

class MerkleTree {
private:
    vector<vector<string>> tree_layers;  // �洢���ĸ���ڵ㣬tree_layers[0]ΪҶ�ӽڵ��

public:
    // ���캯������Ҷ�����ݹ���Merkle��
    MerkleTree(const vector<string>& leaf_nodes) {
        tree_layers.push_back(leaf_nodes);
        construct_tree();
    }

    // ����Merkle��
    void construct_tree() {
        // ��Ҷ�Ӳ㿪ʼ���Ϲ�����ֱ�����ڵ�
        while (tree_layers.back().size() > 1) {
            const vector<string>& prev_layer = tree_layers.back();
            vector<string> current_layer;

            // ÿ�����ڵ�ϲ�Ϊһ�����ڵ�
            for (size_t i = 0; i < prev_layer.size(); i += 2) {
                if (i + 1 < prev_layer.size()) {
                    current_layer.push_back(compute_node_hash(prev_layer[i], prev_layer[i + 1]));
                }
                else {
                    // �����������ڵ����������һ���ڵ�������ϲ�
                    current_layer.push_back(compute_node_hash(prev_layer[i], prev_layer[i]));
                }
            }

            tree_layers.push_back(current_layer);
        }
    }

    // ��ȡ����ϣ
    std::string get_root_hash() const {
        return tree_layers.back()[0];
    }

    // ����ָ������Ҷ�ӽڵ�Ĵ�����֤��
    std::vector<string> generate_proof(size_t index) const {
        vector<string> proof;
        size_t current_index = index;

        // ��Ҷ�Ӳ������ռ�֤���ڵ�
        for (size_t layer = 0; layer < tree_layers.size() - 1; ++layer) {
            size_t sibling_index = (current_index % 2 == 0) ? current_index + 1 : current_index - 1;

            if (sibling_index < tree_layers[layer].size()) {
                proof.push_back(tree_layers[layer][sibling_index]);
            }

            current_index /= 2;
        }

        return proof;
    }

    // ��֤������֤��
    bool validate_proof(const string& leaf_hash, const vector<string>& proof,
        size_t index, const string& root_hash) const {
        string current_hash = leaf_hash;
        size_t current_index = index;

        // �������ϣ����֤�Ƿ������ϣһ��
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

// ====================== �����Ժ��� ======================
int main() {
    const size_t LEAF_COUNT = 100000;  // Ҷ�ӽڵ�����
    vector<string> leaf_hashes;
    leaf_hashes.reserve(LEAF_COUNT);

    // ����Ҷ�ӽڵ��ϣ
    for (size_t i = 0; i < LEAF_COUNT; ++i) {
        leaf_hashes.push_back(compute_leaf_hash("leaf_data_" + to_string(i)));
    }

    // ����Merkle������ʱ
    auto start_time = chrono::high_resolution_clock::now();
    MerkleTree merkle_tree(leaf_hashes);
    auto end_time = chrono::high_resolution_clock::now();

    // ���������Ϣ
    double build_duration = chrono::duration<double>(end_time - start_time).count();
    cout << "[Merkle���������] Ҷ�ӽڵ���=" << LEAF_COUNT
        << " ����ϣ=" << merkle_tree.get_root_hash().substr(0, 16) << "..." << endl;
    cout << "����ʱ��=" << build_duration << " ��" << endl;

    // ���Դ�����֤��
    size_t test_index = 12345;  // ���Ե�Ҷ�ӽڵ�����
    vector<string> inclusion_proof = merkle_tree.generate_proof(test_index);
    bool is_valid = merkle_tree.validate_proof(leaf_hashes[test_index], inclusion_proof,
        test_index, merkle_tree.get_root_hash());

    cout << "��֤Ҷ�ӽڵ� " << test_index << " -> " << (is_valid ? "�ɹ�" : "ʧ��") << endl;

    return 0;
}
