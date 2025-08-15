#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

// 常量定义
#define BLOCK_SIZE_BYTES 16
#define MAX_BLOCKS 1000000

// SM4 算法常量
static const uint32_t CK[32] = {
    0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269,
    0x70777e85, 0x8c939aa1, 0xa8afb6bd, 0xc4cbd2d9,
    0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
    0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9,
    0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229,
    0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
    0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
    0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279
};

static const uint8_t SBOX[256] = {
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48
};

// 循环左移
static inline uint32_t rotate_left_32(uint32_t value, int shift_bits) {
    return (value << shift_bits) | (value >> (32 - shift_bits));
}
//T_op
static inline uint32_t opt_T_transform(uint32_t input_word) {
    return (SBOX[(input_word >> 24) & 0xFF] << 24) |
        (SBOX[(input_word >> 16) & 0xFF] << 16) |
        (SBOX[(input_word >> 8) & 0xFF] << 8) |
        (SBOX[input_word & 0xFF]);
}
//线性变换L
static inline uint32_t linear_transform_l(uint32_t input_value) {
    return input_value ^ rotate_left_32(input_value, 2) ^
        rotate_left_32(input_value, 10) ^ rotate_left_32(input_value, 18) ^
        rotate_left_32(input_value, 24);
}
//T变换
static inline uint32_t composite_transform_t(uint32_t input_word) {
    return linear_transform_l(opt_T_transform(input_word));
}

// SM4 密钥扩展
void generate_round_keys(const uint8_t master_key[16], uint32_t round_keys[32]) {
    static const uint32_t FK[4] = { 0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc };
    uint32_t MK[4];

    for (int i = 0; i < 4; i++) {
        MK[i] = ((uint32_t)master_key[4 * i] << 24) |
            ((uint32_t)master_key[4 * i + 1] << 16) |
            ((uint32_t)master_key[4 * i + 2] << 8) |
            ((uint32_t)master_key[4 * i + 3]);
    }

    uint32_t K[36];
    for (int i = 0; i < 4; i++) {
        K[i] = MK[i] ^ FK[i];
    }

    for (int i = 0; i < 32; i++) {
        uint32_t tmp = opt_T_transform(K[i + 1] ^
            K[i + 2] ^
            K[i + 3] ^
            CK[i]);
        tmp = tmp ^ rotate_left_32(tmp, 13) ^
            rotate_left_32(tmp, 23);
        K[i + 4] = K[i] ^ tmp;
        round_keys[i] = K[i + 4];
    }
}

// 基础加密实现
void sm4_encrypt_basic(const uint8_t input_block[16], uint8_t output_block[16], const uint32_t round_keys[32]) {
    uint32_t state_words[36];

    for (int i = 0; i < 4; i++) {
        state_words[i] = ((uint32_t)input_block[4 * i] << 24) |
            ((uint32_t)input_block[4 * i + 1] << 16) |
            ((uint32_t)input_block[4 * i + 2] << 8) |
            ((uint32_t)input_block[4 * i + 3]);
    }

    for (int i = 0; i < 32; i++) {
        state_words[i + 4] = state_words[i] ^
            composite_transform_t(state_words[i + 1] ^
                state_words[i + 2] ^
                state_words[i + 3] ^
                round_keys[i]);
    }

    for (int i = 0; i < 4; i++) {
        uint32_t output_word = state_words[35 - i];
        output_block[4 * i] = output_word >> 24;
        output_block[4 * i + 1] = output_word >> 16;
        output_block[4 * i + 2] = output_word >> 8;
        output_block[4 * i + 3] = output_word;
    }
}


// 时间测量函数
double get_current_time() {
    using namespace std::chrono;
    return duration<double>(high_resolution_clock::now().time_since_epoch()).count();
}

// T-Table 优化
static uint32_t LOOKUP_TABLES[4][256];

void initialize_t_tables() {
    for (int i = 0; i < 256; i++) {
        uint32_t transformed_value = opt_T_transform(i << 24);
        uint32_t linear_result = transformed_value ^ rotate_left_32(transformed_value, 2) ^
            rotate_left_32(transformed_value, 10) ^
            rotate_left_32(transformed_value, 18) ^
            rotate_left_32(transformed_value, 24);
        LOOKUP_TABLES[0][i] = linear_result;
        LOOKUP_TABLES[1][i] = rotate_left_32(linear_result, 8);
        LOOKUP_TABLES[2][i] = rotate_left_32(linear_result, 16);
        LOOKUP_TABLES[3][i] = rotate_left_32(linear_result, 24);
    }
}

inline uint32_t fast_t_transform(uint32_t input_word) {
    return LOOKUP_TABLES[0][(input_word >> 24) & 0xFF] ^
        LOOKUP_TABLES[1][(input_word >> 16) & 0xFF] ^
        LOOKUP_TABLES[2][(input_word >> 8) & 0xFF] ^
        LOOKUP_TABLES[3][input_word & 0xFF];
}


void sm4_encrypt_t_table(const uint8_t input_block[16], uint8_t output_block[16], const uint32_t round_keys[32]) {
    uint32_t state_words[36];

    for (int i = 0; i < 4; i++) {
        state_words[i] = ((uint32_t)input_block[4 * i] << 24) |
            ((uint32_t)input_block[4 * i + 1] << 16) |
            ((uint32_t)input_block[4 * i + 2] << 8) |
            ((uint32_t)input_block[4 * i + 3]);
    }

    for (int i = 0; i < 32; i++) {
        state_words[i + 4] = state_words[i] ^
            fast_t_transform(state_words[i + 1] ^
                state_words[i + 2] ^
                state_words[i + 3] ^
                round_keys[i]);
    }

    for (int i = 0; i < 4; i++) {
        uint32_t output_word = state_words[35 - i];
        output_block[4 * i] = output_word >> 24;
        output_block[4 * i + 1] = output_word >> 16;
        output_block[4 * i + 2] = output_word >> 8;
        output_block[4 * i + 3] = output_word;
    }
}

void sm4_encrypt4_SIMD(const uint8_t input[64], uint8_t output[64], const uint32_t round_keys[32]) {
    __m256i X0 = _mm256_set_epi32(
        ((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) | ((uint32_t)input[2] << 8) | input[3],
        ((uint32_t)input[16] << 24) | ((uint32_t)input[17] << 16) | ((uint32_t)input[18] << 8) | input[19],
        ((uint32_t)input[32] << 24) | ((uint32_t)input[33] << 16) | ((uint32_t)input[34] << 8) | input[35],
        ((uint32_t)input[48] << 24) | ((uint32_t)input[49] << 16) | ((uint32_t)input[50] << 8) | input[51],
        0, 0, 0, 0);

}

// CTR 模式
void sm4_ctr_encrypt_mode(uint8_t* output_buffer, const uint8_t* input_buffer, size_t block_count,
    const uint32_t round_keys[32], uint8_t initialization_vector[16]) {
    uint8_t counter_block[16];
    memcpy(counter_block, initialization_vector, 16);

    for (size_t i = 0; i < block_count; i++) {
        uint8_t keystream_block[16];
        sm4_encrypt_t_table(counter_block, keystream_block, round_keys);

        for (int j = 0; j < 16; j++) {
            output_buffer[i * 16 + j] = input_buffer[i * 16 + j] ^ keystream_block[j];
        }

        // 计数器递增
        for (int k = 15; k >= 0; k--) {
            if (++counter_block[k]) break;
        }
    }
}


#include <wmmintrin.h>

static inline __m128i ghash_multiplication(__m128i operand_x, __m128i operand_y) {
    return _mm_clmulepi64_si128(operand_x, operand_y, 0x00);
}

void sm4_gcm_encrypt_mode(uint8_t* output_buffer, const uint8_t* input_buffer, size_t block_count,
    const uint32_t round_keys[32], uint8_t initialization_vector[16]) {
    uint8_t hash_subkey[16] = { 0 };
    sm4_encrypt_t_table(hash_subkey, hash_subkey, round_keys);
    __m128i hash_key = _mm_loadu_si128((__m128i*)hash_subkey);
    __m128i authentication_tag = _mm_setzero_si128();

    uint8_t counter_block[16];
    memcpy(counter_block, initialization_vector, 16);

    for (size_t i = 0; i < block_count; i++) {
        uint8_t keystream_block[16];
        sm4_encrypt_t_table(counter_block, keystream_block, round_keys);

        for (int j = 0; j < 16; j++) {
            output_buffer[i * 16 + j] = input_buffer[i * 16 + j] ^ keystream_block[j];
        }

        __m128i data_block = _mm_loadu_si128((__m128i*)(output_buffer + i * 16));
        authentication_tag = _mm_xor_si128(authentication_tag, data_block);
        authentication_tag = ghash_multiplication(authentication_tag, hash_key);

        // 计数器递增
        for (int k = 15; k >= 0; k--) {
            if (++counter_block[k]) break;
        }
    }
}

// 性能测试主函数
int main(int argc, char* argv[]) {
    initialize_t_tables();

    uint8_t encryption_key[16] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10 };
    uint32_t round_keys[32];
    generate_round_keys(encryption_key, round_keys);

    size_t block_count = 100000; 
    if (argc >= 3) block_count = std::stoul(argv[2]);

    std::string test_mode = (argc >= 2) ? argv[1] : "all";

    std::vector<uint8_t> input_data(block_count * 16, 0x11);
    std::vector<uint8_t> output_data(block_count * 16);

    auto benchmark_function = [&](auto test_function, const char* mode_name) {
        auto start_time = std::chrono::high_resolution_clock::now();
        test_function();
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
        double throughput_mbps = (block_count * 16.0 / elapsed_seconds) / 1024 / 1024;
        printf("%-10s | %-10zu | %.2f MB/s\n", mode_name, block_count, throughput_mbps);
        };



    if (test_mode == "base" || test_mode == "all") {
        benchmark_function([&] {
            for (size_t i = 0; i < block_count; i++) {
                sm4_encrypt_basic(&input_data[i * 16], &output_data[i * 16], round_keys);
            }
            }, "Base");
    }

    if (test_mode == "ttable" || test_mode == "all") {
        benchmark_function([&] {
            for (size_t i = 0; i < block_count; i++) {
                sm4_encrypt_t_table(&input_data[i * 16], &output_data[i * 16], round_keys);
            }
            }, "T-table");
    }

    if (test_mode == "gcm" || test_mode == "all") {
        uint8_t initialization_vector[16] = { 0 };
        benchmark_function([&] {
            sm4_gcm_encrypt_mode(output_data.data(), input_data.data(), block_count, round_keys, initialization_vector);
            }, "GCM");
    }

    return 0;
}
