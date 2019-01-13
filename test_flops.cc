#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <stdlib.h>

struct benchmark
{
    virtual int get_flops_per_iteration() const = 0;
    virtual int get_independent_instructions() const = 0;
    virtual float run_kernel(size_t iterations) const = 0;

    const __m256 mul0 = _mm256_set1_ps(1.001f);
    const __m256 mul1 = _mm256_set1_ps(1.002f);
};

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

struct benchmark_12 : benchmark
{

    int get_independent_instructions() const override
    {
        return 12;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 12 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r4 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r5 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r6 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r7 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r8 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r9 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 rA = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 rB = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);
                r8 = _mm256_fmadd_ps(mul0, mul1, r8);
                r9 = _mm256_fmadd_ps(mul0, mul1, r9);
                rA = _mm256_fmadd_ps(mul0, mul1, rA);
                rB = _mm256_fmadd_ps(mul0, mul1, rB);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r6);
            r1 = _mm256_add_ps(r1, r7);
            r2 = _mm256_add_ps(r2, r8);
            r3 = _mm256_add_ps(r3, r9);
            r4 = _mm256_add_ps(r4, rA);
            r5 = _mm256_add_ps(r5, rB);

            r0 = _mm256_add_ps(r0, r3);
            r1 = _mm256_add_ps(r1, r4);
            r2 = _mm256_add_ps(r2, r5);

            r0 = _mm256_add_ps(r0, r1);
            r0 = _mm256_add_ps(r0, r2);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_11 : benchmark
{

    int get_independent_instructions() const override
    {
        return 11;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 11 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r4 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r5 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r6 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r7 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r8 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r9 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 rA = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);
                r8 = _mm256_fmadd_ps(mul0, mul1, r8);
                r9 = _mm256_fmadd_ps(mul0, mul1, r9);
                rA = _mm256_fmadd_ps(mul0, mul1, rA);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r5);
            r1 = _mm256_add_ps(r1, r6);
            r2 = _mm256_add_ps(r2, r7);
            r3 = _mm256_add_ps(r3, r8);
            r4 = _mm256_add_ps(r4, r9);



            r0 = _mm256_add_ps(r0, rA);
            r1 = _mm256_add_ps(r1, r3);
            r2 = _mm256_add_ps(r2, r4);

            r1 = _mm256_add_ps(r1, r2);
            r0 = _mm256_add_ps(r0, r1);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_10 : benchmark
{

    int get_independent_instructions() const override
    {
        return 10;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 10 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r4 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r5 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r6 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r7 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r8 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r9 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);
                r8 = _mm256_fmadd_ps(mul0, mul1, r8);
                r9 = _mm256_fmadd_ps(mul0, mul1, r9);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r5);
            r1 = _mm256_add_ps(r1, r6);
            r2 = _mm256_add_ps(r2, r7);
            r3 = _mm256_add_ps(r3, r8);
            r4 = _mm256_add_ps(r4, r9);

            r0 = _mm256_add_ps(r0, r2);
            r1 = _mm256_add_ps(r1, r3);

            r0 = _mm256_add_ps(r0, r4);

            r0 = _mm256_add_ps(r0, r1);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_9 : benchmark
{
    int get_independent_instructions() const override
    {
        return 9;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 9 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r4 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r5 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r6 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r7 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r8 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);
                r7 = _mm256_fmadd_ps(mul0, mul1, r7);
                r8 = _mm256_fmadd_ps(mul0, mul1, r8);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r4);
            r1 = _mm256_add_ps(r1, r5);
            r2 = _mm256_add_ps(r2, r6);
            r3 = _mm256_add_ps(r3, r7);

            r0 = _mm256_add_ps(r0, r1);
            r2 = _mm256_add_ps(r2, r3);
            
            r0 = _mm256_add_ps(r0, r8);
            r0 = _mm256_add_ps(r0, r2);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_8 : benchmark
{

    int get_independent_instructions() const override
    {
        return 8;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 8 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
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

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r4);
            r1 = _mm256_add_ps(r1, r5);
            r2 = _mm256_add_ps(r2, r6);
            r3 = _mm256_add_ps(r3, r7);

            r0 = _mm256_add_ps(r0, r1);
            r2 = _mm256_add_ps(r2, r3);

            r0 = _mm256_add_ps(r0, r2);
            return __m256_reduce_add_ps(r0);
    }
};

struct benchmark_7 : benchmark
{
    int get_independent_instructions() const override
    {
        return 7;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 7 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r4 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r5 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r6 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());

            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);
                r6 = _mm256_fmadd_ps(mul0, mul1, r6);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r3);
            r1 = _mm256_add_ps(r1, r4);
            r2 = _mm256_add_ps(r2, r5);
            
            r0 = _mm256_add_ps(r0, r6);
            r1 = _mm256_add_ps(r1, r2);

            r0 = _mm256_add_ps(r0, r1);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_6 : benchmark
{

    int get_independent_instructions() const override
    {
        return 6;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 6 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r4 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r5 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);
                r5 = _mm256_fmadd_ps(mul0, mul1, r5);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r3);
            r1 = _mm256_add_ps(r1, r4);
            r2 = _mm256_add_ps(r3, r5);

            r0 = _mm256_add_ps(r0, r1);
            r0 = _mm256_add_ps(r0, r2);

            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_5 : benchmark
{

    int get_independent_instructions() const override
    {
        return 5;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 5 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r4 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);
                r4 = _mm256_fmadd_ps(mul0, mul1, r4);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r2);
            r1 = _mm256_add_ps(r1, r3);
            r0 = _mm256_add_ps(r0, r4);
            
            r0 = _mm256_add_ps(r0, r1);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_4 : benchmark
{

    int get_independent_instructions() const override
    {
        return 4;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 4 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r3 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);
                r3 = _mm256_fmadd_ps(mul0, mul1, r3);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r2);
            r1 = _mm256_add_ps(r1, r3);

            r0 = _mm256_add_ps(r0, r1);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_3 : benchmark
{

    int get_independent_instructions() const override
    {
        return 3;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 3 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r2 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);
                r2 = _mm256_fmadd_ps(mul0, mul1, r2);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r1);
            r0 = _mm256_add_ps(r0, r2);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_2 : benchmark
{
    int get_independent_instructions() const override
    {
        return 2;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 2 * 2 * 8;
    }
    float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
                r1 = _mm256_fmadd_ps(mul0, mul1, r1);

            }while (--iterations);

            r0 = _mm256_add_ps(r0, r1);
            return __m256_reduce_add_ps(r0);
    }
};
struct benchmark_1 : benchmark
{
    int get_independent_instructions() const override
    {
        return 1;
    }
    int get_flops_per_iteration() const override
    {
        // OPERATIONS x INSTRUCTIONS x SIZE
        return 1 * 2 * 8;
    }
        float run_kernel(size_t iterations) const override
    {
            __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
            do{
                r0 = _mm256_fmadd_ps(mul0, mul1, r0);
            }while (--iterations);

            return __m256_reduce_add_ps(r0);
    }
};

void bench_kernel(const benchmark &bench)
{
    double seconds = 1;
    int m_block_size = 10000000;
    int m_flops_per_iteration = bench.get_flops_per_iteration();

    std::chrono::duration<double> duration(seconds);
    auto start = std::chrono::system_clock::now();
    float sum = 0;
    uint64_t iterations = 0;
    std::chrono::duration<double> clock;
    do
    {
        sum += bench.run_kernel(m_block_size);
        iterations++;
        clock = std::chrono::system_clock::now() - start;
    }
    while (clock < duration);

    double flops = iterations * m_flops_per_iteration * m_block_size / seconds;
    std::cout << "\n";
    std::cout << "Ind Instructions: " << bench.get_independent_instructions() << std::endl;
    std::cout << "Iterations:       " << iterations << std::endl;
    std::cout << "Blocksize:        " << m_block_size << std::endl;
    std::cout << "m_flops_per_iter: " << m_flops_per_iteration << std::endl;
    std::cout << "GFlops:           " << flops / 1000000000 << std::endl;
    std::cout << "Flops:            " << flops << std::endl;
    std::cout << "SUM:              " << sum << std::endl;

}

void runAll()
{
    bench_kernel(benchmark_1 {});
    bench_kernel(benchmark_2 {});
    bench_kernel(benchmark_3 {});
    bench_kernel(benchmark_4 {});
    bench_kernel(benchmark_5 {});
    bench_kernel(benchmark_6 {});
    bench_kernel(benchmark_7 {});
    bench_kernel(benchmark_8 {});
    bench_kernel(benchmark_9 {});
    bench_kernel(benchmark_10 {});
    bench_kernel(benchmark_11 {});
    bench_kernel(benchmark_12 {});
}
int main(int argc, char* argv[])
{
    if(argc >= 2)
    {
        for(int i=0; i<argc; i++)
        {
            int ix = std::atoi(argv[i]);
            switch(ix)
            {
                case 1: {bench_kernel(benchmark_1{}); break;}
                case 2: {bench_kernel(benchmark_2{}); break;}
                case 3: {bench_kernel(benchmark_3{}); break;}
                case 4: {bench_kernel(benchmark_4{}); break;}
                case 5: {bench_kernel(benchmark_5{}); break;}
                case 6: {bench_kernel(benchmark_6{}); break;}
                case 7: {bench_kernel(benchmark_7{}); break;}
                case 8: {bench_kernel(benchmark_8{}); break;}
                case 9: {bench_kernel(benchmark_9{}); break;}
                case 10: {bench_kernel(benchmark_10{}); break;}
                case 11: {bench_kernel(benchmark_11{}); break;}
                case 12: {bench_kernel(benchmark_12{}); break;}
             
            }
        }
    }
    else
    {
        runAll();
    }
}

/*

*/