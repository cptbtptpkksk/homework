#include <immintrin.h>
#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <future>
#include <sstream>

// 配置参数
#define MESSAGE_BLOCK_SIZE 64
#define HASH_OUTPUT_SIZE 32
#define SIMD_VECTOR_WIDTH 8  // AVX2: 256-bit, 8个32-bit字

// SM3算法初始向量
const uint32_t IV[8] = {
    0x7380166F,
    0x4914B2B9,
    0x172442D7,
    0xDA8A0600,
    0xA96F30BC,
    0x163138AA,
    0xE38DEE4D,
    0xB0FB0E4E
};

// T_j常量表
const uint32_t T[64] = {
    0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,
    0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,0x79CC4519,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,
    0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A,0x7A879D8A
};

// SIMD 工具函数 (保持与原始代码一致)
inline __m256i ROTL32(__m256i x, int n) {
    return _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - n));
}

inline __m256i P0(__m256i x) {
    return _mm256_xor_si256(_mm256_xor_si256(x, ROTL32(x, 9)), ROTL32(x, 17));
}

inline __m256i P1(__m256i x) {
    return _mm256_xor_si256(_mm256_xor_si256(x, ROTL32(x, 15)), ROTL32(x, 23));
}

inline __m256i FF(__m256i X, __m256i Y, __m256i Z, int j) {
    return (j < 16) ? _mm256_xor_si256(_mm256_xor_si256(X, Y), Z)
        : _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(X, Y), _mm256_and_si256(X, Z)), _mm256_and_si256(Y, Z));
}

inline __m256i GG(__m256i X, __m256i Y, __m256i Z, int j) {
    return (j < 16) ? _mm256_xor_si256(_mm256_xor_si256(X, Y), Z)
        : _mm256_or_si256(_mm256_and_si256(X, Y), _mm256_and_si256(_mm256_andnot_si256(X, _mm256_set1_epi32(-1)), Z));
}

// 增强的SM3哈希处理器类
class EnhancedSM3Processor {
private:
    // 消息扩展处理
    void process_message_expansion(uint8_t data_blocks[SIMD_VECTOR_WIDTH][MESSAGE_BLOCK_SIZE],
        __m256i* expanded_words, __m256i* derived_words) {
        // 初始16个字处理
        for (int word_idx = 0; word_idx < 16; word_idx++) {
            uint32_t temp_values[SIMD_VECTOR_WIDTH];
            for (int simd_idx = 0; simd_idx < SIMD_VECTOR_WIDTH; simd_idx++) {
                temp_values[simd_idx] = ((uint32_t)data_blocks[simd_idx][4 * word_idx] << 24) |
                    ((uint32_t)data_blocks[simd_idx][4 * word_idx + 1] << 16) |
                    ((uint32_t)data_blocks[simd_idx][4 * word_idx + 2] << 8) |
                    ((uint32_t)data_blocks[simd_idx][4 * word_idx + 3]);
            }
            expanded_words[word_idx] = _mm256_loadu_si256((__m256i*)temp_values);
        }

        // 扩展到68个字
        for (int word_idx = 16; word_idx < 68; word_idx++) {
            expanded_words[word_idx] = _mm256_xor_si256(
                P1(_mm256_xor_si256(_mm256_xor_si256(expanded_words[word_idx - 16], expanded_words[word_idx - 9]),
                    ROTL32(expanded_words[word_idx - 3], 15))),
                _mm256_xor_si256(ROTL32(expanded_words[word_idx - 13], 7), expanded_words[word_idx - 6])
            );
        }

        // 生成64个W'字
        for (int word_idx = 0; word_idx < 64; word_idx++) {
            derived_words[word_idx] = _mm256_xor_si256(expanded_words[word_idx], expanded_words[word_idx + 4]);
        }
    }

    // 压缩函数核心处理
    void execute_compression_rounds(__m256i& state_a, __m256i& state_b, __m256i& state_c, __m256i& state_d,
        __m256i& state_e, __m256i& state_f, __m256i& state_g, __m256i& state_h,
        const __m256i* expanded_words, const __m256i* derived_words) {
        // 64轮迭代压缩
        for (int round_idx = 0; round_idx < 64; round_idx++) {
            __m256i round_constant = _mm256_set1_epi32(T[round_idx]);
            __m256i ss1 = ROTL32(_mm256_add_epi32(_mm256_add_epi32(ROTL32(state_a, 12), state_e),
                ROTL32(round_constant, round_idx % 32)), 7);
            __m256i ss2 = _mm256_xor_si256(ss1, ROTL32(state_a, 12));
            __m256i tt1 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(FF(state_a, state_b, state_c, round_idx), state_d), ss2), derived_words[round_idx]);
            __m256i tt2 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(GG(state_e, state_f, state_g, round_idx), state_h), ss1), expanded_words[round_idx]);

            // 状态更新
            state_d = state_c;
            state_c = ROTL32(state_b, 9);
            state_b = state_a;
            state_a = tt1;
            state_h = state_g;
            state_g = ROTL32(state_f, 19);
            state_f = state_e;
            state_e = P0(tt2);
        }
    }

    // 结果合并处理
    void merge_final_results(const __m256i& final_a, const __m256i& final_b, const __m256i& final_c, const __m256i& final_d,
        const __m256i& final_e, const __m256i& final_f, const __m256i& final_g, const __m256i& final_h,
        uint32_t result_digest[SIMD_VECTOR_WIDTH][8]) {
        uint32_t buffer[8][SIMD_VECTOR_WIDTH];
        _mm256_storeu_si256((__m256i*)buffer[0], final_a);
        _mm256_storeu_si256((__m256i*)buffer[1], final_b);
        _mm256_storeu_si256((__m256i*)buffer[2], final_c);
        _mm256_storeu_si256((__m256i*)buffer[3], final_d);
        _mm256_storeu_si256((__m256i*)buffer[4], final_e);
        _mm256_storeu_si256((__m256i*)buffer[5], final_f);
        _mm256_storeu_si256((__m256i*)buffer[6], final_g);
        _mm256_storeu_si256((__m256i*)buffer[7], final_h);

        for (int simd_idx = 0; simd_idx < SIMD_VECTOR_WIDTH; simd_idx++) {
            for (int state_idx = 0; state_idx < 8; state_idx++) {
                result_digest[simd_idx][state_idx] = buffer[state_idx][simd_idx] ^ IV[state_idx];
            }
        }
    }

public:
    // SIMD压缩函数 - 处理8个消息块
    void process_simd_compression(uint8_t input_blocks[SIMD_VECTOR_WIDTH][MESSAGE_BLOCK_SIZE],
        uint32_t output_digest[SIMD_VECTOR_WIDTH][8]) {
        // 初始化状态向量
        __m256i state_a = _mm256_set1_epi32(IV[0]);
        __m256i state_b = _mm256_set1_epi32(IV[1]);
        __m256i state_c = _mm256_set1_epi32(IV[2]);
        __m256i state_d = _mm256_set1_epi32(IV[3]);
        __m256i state_e = _mm256_set1_epi32(IV[4]);
        __m256i state_f = _mm256_set1_epi32(IV[5]);
        __m256i state_g = _mm256_set1_epi32(IV[6]);
        __m256i state_h = _mm256_set1_epi32(IV[7]);

        __m256i expanded_words[68], derived_words[64];

        // 执行消息扩展
        process_message_expansion(input_blocks, expanded_words, derived_words);

        // 执行压缩轮次
        execute_compression_rounds(state_a, state_b, state_c, state_d, state_e, state_f, state_g, state_h,
            expanded_words, derived_words);

        // 合并最终结果
        merge_final_results(state_a, state_b, state_c, state_d, state_e, state_f, state_g, state_h, output_digest);
    }
};

// 多线程并行处理器
class ParallelSM3Executor {
private:
    EnhancedSM3Processor processor_;
    int thread_count_;

public:
    ParallelSM3Executor(int threads = 0) : thread_count_(threads) {
        if (thread_count_ <= 0) {
            thread_count_ = std::thread::hardware_concurrency();
        }
    }

    // 并行处理消息批次
    void execute_parallel_processing(const std::vector<std::vector<uint8_t>>& message_batch) {
        int batch_size = SIMD_VECTOR_WIDTH;
        int total_messages = message_batch.size();
        int messages_per_thread = (total_messages + thread_count_ - 1) / thread_count_;

        auto thread_worker = [&](int start_pos, int end_pos) {
            uint8_t local_blocks[SIMD_VECTOR_WIDTH][MESSAGE_BLOCK_SIZE];
            uint32_t local_digest[SIMD_VECTOR_WIDTH][8];

            for (int msg_idx = start_pos; msg_idx < end_pos; msg_idx += batch_size) {
                int current_batch = std::min(batch_size, end_pos - msg_idx);

                // 准备当前批次的数据
                for (int batch_idx = 0; batch_idx < current_batch; batch_idx++) {
                    memcpy(local_blocks[batch_idx], message_batch[msg_idx + batch_idx].data(), MESSAGE_BLOCK_SIZE);
                }

                // 执行SIMD压缩
                processor_.process_simd_compression(local_blocks, local_digest);
            }
            };

        // 创建并启动工作线程
        std::vector<std::future<void>> thread_futures;
        for (int thread_idx = 0; thread_idx < thread_count_; thread_idx++) {
            int thread_start = thread_idx * messages_per_thread;
            int thread_end = std::min(thread_start + messages_per_thread, total_messages);
            if (thread_start < thread_end) {
                thread_futures.emplace_back(std::async(std::launch::async, thread_worker, thread_start, thread_end));
            }
        }

        // 等待所有线程完成
        for (auto& future : thread_futures) {
            future.wait();
        }
    }

    int get_thread_count() const { return thread_count_; }
};

// 消息生成器
class MessageGenerator {
public:
    // 生成填充后的消息块
    static std::vector<uint8_t> create_padded_message(size_t content_length) {
        std::vector<uint8_t> message(MESSAGE_BLOCK_SIZE, 0);

        // 填充内容
        for (size_t i = 0; i < content_length; i++) {
            message[i] = 'a' + (i % 26);
        }

        // 添加填充位
        message[content_length] = 0x80;

        // 添加长度信息
        uint64_t bit_length = content_length * 8;
        for (int i = 0; i < 8; i++) {
            message[MESSAGE_BLOCK_SIZE - 1 - i] = (uint8_t)(bit_length >> (i * 8));
        }

        return message;
    }

    // 生成测试消息批次
    static std::vector<std::vector<uint8_t>> generate_test_batch(int batch_size, size_t message_length) {
        std::vector<std::vector<uint8_t>> batch;
        batch.reserve(batch_size);

        for (int i = 0; i < batch_size; i++) {
            batch.push_back(create_padded_message(message_length));
        }

        return batch;
    }
};

// 性能测试器
class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        int batch_size;
        size_t message_length;
        int thread_count;
        double execution_time;
        double throughput_mbps;
    };

    static BenchmarkResult run_benchmark(int batch_size, size_t message_length, int thread_count) {
        auto message_batch = MessageGenerator::generate_test_batch(batch_size, message_length);
        ParallelSM3Executor executor(thread_count);

        auto start_time = std::chrono::high_resolution_clock::now();
        executor.execute_parallel_processing(message_batch);
        auto end_time = std::chrono::high_resolution_clock::now();

        double execution_seconds = std::chrono::duration<double>(end_time - start_time).count();
        double throughput = (batch_size * message_length) / (1024.0 * 1024.0) / execution_seconds;

        return { batch_size, message_length, thread_count, execution_seconds, throughput };
    }

    static void print_benchmark_result(const BenchmarkResult& result) {
        std::cout << "批次大小: " << result.batch_size << ", 消息长度: " << result.message_length << " 字节\n";
        std::cout << "线程数: " << result.thread_count << ", 执行时间: " << result.execution_time << " 秒\n";
        std::cout << "吞吐量: " << std::fixed << std::setprecision(2) << result.throughput_mbps << " MB/s\n";
    }
};

int main() {
    std::cout << "增强版SM3 SIMD多线程性能测试 \n\n";

    // 配置参数
    int test_batch_size = 8000;
    size_t test_message_length = 32;
    int test_thread_count = std::thread::hardware_concurrency();

    // 执行性能测试
    auto benchmark_result = PerformanceBenchmark::run_benchmark(
        test_batch_size, test_message_length, test_thread_count);

    // 输出结果
    PerformanceBenchmark::print_benchmark_result(benchmark_result);

    std::cout << "\n 测试完成 \n";

    return 0;
}
