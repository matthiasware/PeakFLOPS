#include <iostream>
#include <immintrin.h>
#include <chrono>

static inline float __m256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

struct benchmark
{
    const __m256 mul0 = _mm256_set1_ps(1.001f);
    const __m256 mul1 = _mm256_set1_ps(1.002f);

    // BLOCKS x OPERATIONS x INSTRUCTIONS x SIZE
    const int flops_per_iteration = 14 * 8 * 2 * 8;

    const int independent_instructions = 8;

    __m256 run_kernel(size_t iterations)
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r4 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r5 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r6 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r7 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fnmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fnmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fnmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fnmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fnmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fnmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fnmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fnmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fnmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fnmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fnmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fnmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fnmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fnmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fnmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fnmadd_ps(mul0, mul1, r7);


                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fnmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fnmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fnmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fnmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fnmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fnmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fnmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fnmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fnmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fnmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fnmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fnmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fnmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fnmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fnmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fnmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fnmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fnmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fnmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fnmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fnmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fnmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fnmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fnmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fnmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fnmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fnmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fnmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fnmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fnmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fnmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fnmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);

                r0 = _mm256_fnmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fnmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fnmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fnmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fnmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fnmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fnmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fnmadd_ps(mul0, mul1, r7);
            }while (--iterations);

            r0 = _mm256_add_ps(r0, r4);
            r1 = _mm256_add_ps(r1, r5);
            r2 = _mm256_add_ps(r2, r6);
            r3 = _mm256_add_ps(r3, r7);

            r0 = _mm256_add_ps(r0, r1);
            r2 = _mm256_add_ps(r2, r3);

            r0 = _mm256_add_ps(r0, r2);
            return r0;
    }
};

int main(int argc, char* argv[])
{
    benchmark bench;

    double seconds = 1;
    int m_block_size = 1000000;

    __m256 sum = _mm256_set1_ps(0.0f);
    float fsum = 0;
    uint64_t iterations = 0;

    std::chrono::duration<double> run_duration(seconds);
    std::chrono::duration<double> measure_duration(0);
    std::chrono::duration<double> clock;

    std::chrono::time_point<std::chrono::system_clock> start_iter;
    std::chrono::time_point<std::chrono::system_clock> end_iter;
    
    auto start = std::chrono::system_clock::now();
    do
    {
        start_iter = std::chrono::system_clock::now();
        sum = bench.run_kernel(m_block_size);
        end_iter = std::chrono::system_clock::now();
        fsum += __m256_reduce_add_ps(sum);
        measure_duration += end_iter - start_iter;
        clock = end_iter - start;
        iterations++;
    }
    while (clock < run_duration);

    std::cout << "============= COFIG =================" << std::endl;
    std::cout << "Runtime measure (sec): " << run_duration.count() << std::endl;
    std::cout << "Blocksize:             " << m_block_size << std::endl;
    std::cout << "Ind Instructions:      " << bench.independent_instructions << std::endl;
    std::cout << "FLOPS / iteration:     " << bench.flops_per_iteration << std::endl;

    std::cout << "============= Benchmark =============" << std::endl;
    double flops = iterations * bench.flops_per_iteration * m_block_size / measure_duration.count();
    std::cout << "Runtime calc.:    " << fsum << std::endl;
    std::cout << "Iterations:       " << iterations << std::endl;
    std::cout << "Flops:            " << flops << std::endl;
    std::cout << "GFlops:           " << flops / 1000000000 << std::endl;
}

/*

*/