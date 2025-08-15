#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define BLOCK_SIZE_BYTES 16
#define KEY_SIZE_BYTES 16
#define ROUND_COUNT 32

// 系统参数常量
static const uint32_t FK[4] = {
    0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc
};

// 轮常量
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

// 字节替换表
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

// 预计算查找表
static uint32_t T_table[4][256];

// 循环左移操作
static inline uint32_t rotate_left_32(uint32_t value, int shift_bits) {
    return (value << shift_bits) | (value >> (32 - shift_bits));
}

// 线性变换L
static uint32_t linear_transform_l(uint32_t input_value) {
    return input_value ^ rotate_left_32(input_value, 2) ^
        rotate_left_32(input_value, 10) ^ rotate_left_32(input_value, 18) ^
        rotate_left_32(input_value, 24);
}

// 初始化T表
static void initialize_T_tables() {
    for (int i = 0; i < 256; i++) {
        uint8_t substituted_byte = SBOX[i];
        uint32_t expanded_value = ((uint32_t)substituted_byte << 24) |
            ((uint32_t)substituted_byte << 16) |
            ((uint32_t)substituted_byte << 8) |
            substituted_byte;
        uint32_t transformed_value = linear_transform_l(expanded_value);

        T_table[0][i] = transformed_value;
        T_table[1][i] = rotate_left_32(transformed_value, 8);
        T_table[2][i] = rotate_left_32(transformed_value, 16);
        T_table[3][i] = rotate_left_32(transformed_value, 24);
    }
}

// 优化的T变换
static inline uint32_t optimized_t_transform(uint32_t input_word) {
    return T_table[0][(input_word >> 24) & 0xFF] ^
        T_table[1][(input_word >> 16) & 0xFF] ^
        T_table[2][(input_word >> 8) & 0xFF] ^
        T_table[3][input_word & 0xFF];
}

// 密钥扩展（未优化版本）
static uint32_t composite_transform_t_1(uint32_t input_word) {
    uint8_t byte_array[4];
    byte_array[0] = (input_word >> 24) & 0xFF;
    byte_array[1] = (input_word >> 16) & 0xFF;
    byte_array[2] = (input_word >> 8) & 0xFF;
    byte_array[3] = input_word & 0xFF;

    for (int i = 0; i < 4; i++) {
        byte_array[i] = SBOX[byte_array[i]];
    }

    uint32_t intermediate_result = (byte_array[0] << 24) | (byte_array[1] << 16) |
        (byte_array[2] << 8) | byte_array[3];
    return intermediate_result ^ rotate_left_32(intermediate_result, 13) ^
        rotate_left_32(intermediate_result, 23);
}

static void expand_key(const uint8_t* master_key, uint32_t round_keys[32]) {
    uint32_t master_key_words[4];
    for (int i = 0; i < 4; i++) {
        master_key_words[i] = ((uint32_t)master_key[4 * i] << 24) |
            ((uint32_t)master_key[4 * i + 1] << 16) |
            ((uint32_t)master_key[4 * i + 2] << 8) |
            master_key[4 * i + 3];
    }

    uint32_t key_schedule[36];
    for (int i = 0; i < 4; i++) {
        key_schedule[i] = master_key_words[i] ^ FK[i];
    }

    for (int i = 0; i < 32; i++) {
        key_schedule[i + 4] = key_schedule[i] ^
            composite_transform_t_1(key_schedule[i + 1] ^
                key_schedule[i + 2] ^
                key_schedule[i + 3] ^
                CK[i]);
        round_keys[i] = key_schedule[i + 4];
    }
}

// 加密函数（T-table优化版本）
void sm4_encrypt_optimized(const uint8_t input_block[16], uint8_t output_block[16], const uint32_t round_keys[32]) {
    uint32_t state_words[36];

    for (int i = 0; i < 4; i++) {
        state_words[i] = ((uint32_t)input_block[4 * i] << 24) |
            ((uint32_t)input_block[4 * i + 1] << 16) |
            ((uint32_t)input_block[4 * i + 2] << 8) |
            input_block[4 * i + 3];
    }

    for (int i = 0; i < 32; i++) {
        state_words[i + 4] = state_words[i] ^
            optimized_t_transform(state_words[i + 1] ^
                state_words[i + 2] ^
                state_words[i + 3] ^
                round_keys[i]);
    }

    for (int i = 0; i < 4; i++) {
        output_block[4 * i] = (state_words[35 - i] >> 24) & 0xFF;
        output_block[4 * i + 1] = (state_words[35 - i] >> 16) & 0xFF;
        output_block[4 * i + 2] = (state_words[35 - i] >> 8) & 0xFF;
        output_block[4 * i + 3] = state_words[35 - i] & 0xFF;
    }
}

// 主函数
int main() {
    initialize_T_tables();

    uint8_t encryption_key[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };
    uint8_t plaintext_block[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
        0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
    };
    uint8_t ciphertext_block[16];
    uint32_t round_keys[32];

    expand_key(encryption_key, round_keys);
    sm4_encrypt_optimized(plaintext_block, ciphertext_block, round_keys);

    printf("密文: ");
    for (int i = 0; i < 16; i++) printf("%02x", ciphertext_block[i]);
    printf("\n");
    return 0;
}
