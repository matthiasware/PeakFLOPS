# Theoretical Peak FLOPS
Contains code to reach the theoretical Peak FLOPS for the AVX2 instruction set on the following architectures:
Haswell - Broadwell - Skylake - Kaby Lake - Coffee Lake - Whiskey Lake - Amber Lake.

## Requirements
- g++
- intel

## Run
```sh
make
./peak_flops
```

## Calculating the Theoretical Peak FLOPS
The Theoretical Peak FLOPS per core for a given operation can be calculated via:

```math
flops \ second \ core =   flops \ operation 
                        x operations \ instruction 
                        x instructions \ cycle 
                        x  cycles \ second 
```

You can read in [1] what the indiviual factors mean. In order to maximizing the Peak FLOPS, we need to maximize the individual factors.

The operation that maximizes the ```flops \ operation``` factor, is the fused multiply add operation. It performs an addition and an multiplication in one operation. To maximize the ```operations \ instruction``` we need to utilize the vector registers. Each 256 Bit registers can hold 8 32Bit single precision floating point numbers. Maximizing this factor can be achieved by using  the ```_mm256_fmadd_ps``` intrinsic instruction, which executes 8 fused multiply add operations at once. Maximizing the ```instructions \ cycle``` factor means to maximize the instruction throuput of the CPU. On the micro-architectures given above we have two execution units, where each unit executes the ```_mm256_fmadd_ps``` instruction simultaniously if independent. For the last factor ```cylces \ second``` we can use the turbo-boost specification of the specific processor.

For the Intel Core i7-7500U this yield Theoretical Peak FLOPS of ```2 x 8 x 2 x 3.5GHz = 112.0 GFLOPS ``` per core.
Refer to [3] and [4] for further information.

#### Practically Getting There
For practically getting there we have to consider one more thing: Instruction latency.

As mentioned before, the througput for the ```_mm256_fmadd_ps``` instruction is 2 (see [3]) on independent calculations.
The latency for each ```_mm256_fmadd_ps``` is 4 cycles (see [3]). In order to maximize our experimental PEAK Flops, we want to start 2 independent instructions per cycle for each execution unit. This means we need 4 x 2 = 8 independent operations.


E.g. in the following code we have one independent operation:
```c++
float run_kernel(size_t iterations)
{
  __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
  do{
    r0 = _mm256_fmadd_ps(mul0, mul1, r0); // operations 1
  }while (--iterations);
  return __m256_reduce_add_ps(r0);
 }
```
By masureing its performance we get 13.6 GFLOPS. The problem here is, that here we have a series of dependent ```_mm256_fmadd_ps``` operations and we utilize only one of our two execution units.

In order to use our second execution unit we can add a independent ```_mm256_fmadd_ps``` instruction:

```c++
float run_kernel(size_t iterations)
{
        __m256 r0 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
        __m256 r1 = _mm256_set1_ps((float) __builtin_ia32_rdtsc());
        do{
            r0 = _mm256_fmadd_ps(mul0, mul1, r0); // operation 1
            r1 = _mm256_fmadd_ps(mul0, mul1, r1); // operation 2

        }while (--iterations);

        r0 = _mm256_add_ps(r0, r1);
        return __m256_reduce_add_ps(r0);
}
```
Measuring its performance yields 27.2GFLOPS.

With 8 independent iterations we measure 109.6 GFLOPS:
```c++
float run_kernel(size_t iterations)
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
            r0 = _mm256_fmadd_ps(mul0, mul1, r0); // operation 1
            r1 = _mm256_fmadd_ps(mul0, mul1, r1); // operation 2
            r2 = _mm256_fmadd_ps(mul0, mul1, r2); // operation 3
            r3 = _mm256_fmadd_ps(mul0, mul1, r3); // operation 4
            r4 = _mm256_fmadd_ps(mul0, mul1, r4); // operation 5
            r5 = _mm256_fmadd_ps(mul0, mul1, r5); // operation 6
            r6 = _mm256_fmadd_ps(mul0, mul1, r6); // operation 7
            r7 = _mm256_fmadd_ps(mul0, mul1, r7); // operation 8

        }while (--iterations);

        r0 = _mm256_add_ps(r0, r4);
        r1 = _mm256_add_ps(r1, r5);
        r2 = _mm256_add_ps(r2, r6);
        r3 = _mm256_add_ps(r3, r7);

        r0 = _mm256_add_ps(r0, r1);
        r2 = _mm256_add_ps(r2, r3);

        r0 = _mm256_add_ps(r0, r2);
        return sum8(r0);
}
```

You can try it out:


### Validation
Since compilers are smart and try to optimize, it is highly advised to check the assembly code and use a profiler like Intel VTune Amplifier to verify that the operations are performed.

## Resources:
##### [1] Theoretical Peak FLOPS per instruction set on modern Intel CPUs - http://www.dolbeau.name/dolbeau/publications/peak.pdf
##### [2] How to get peak FLOPS - https://www.eidos.ic.i.u-tokyo.ac.jp/~tau/lecture/parallel_distributed/2018/slides/pdf/peak_cpu.pdf
##### [3] Agner Instruction Tables - https://www.agner.org/optimize/instruction_tables.pdf
##### [4] Agner Microarchitecture -  https://www.agner.org/optimize/microarchitecture.pdf 


