# Maximizing FLOPS theoretically and practically

The peak performance of a CPU is often measured in __FLOPS( Floating Point Operations per Second)__. Calculating the theoretical peak performance of a given CPU is often straightforward, but practically maximizing the performed FLOPS is much harder. Here, we try to reach the theoretical peak performance practically for the AVX2 instruction set. This works on the following Intel architectures:

	Haswell - Broadwell - Skylake - Kaby Lake - Coffee Lake - Whiskey Lake - Amber Lake.

##### Requirements
- g++
- clang++

##### Run
```sh
git clone https://github.com/matthiasware/PeakFLOPS.git
mkdir build && cd build
cmake .. && make
./test_flops
```

Depending on your microarchitecture, the operations performed will peak between 8 and 10 independent instructions.

### Theoretical Peak FLOPS

The theoretical peak FLOPS per CPU core can be calculated by maximizing the RHS of the following equation:

```math
FLOPS / core =   flops / operation 
               x operations / instruction 
               x instructions / cycle 
               x cycles / second 
```

Refer to [1] for a detailed explanation. Maximizing the FLOPS requires maximizing the individual factors of the equation above.

The operation that maximizes the ```flops / operation``` factor (__flops__ = floating point operations), is the fused multiply-add operation (see [3]), which performs an addition and a multiplication in one operation: ```a <- a + (b * c)```. This factor is 2/1.

In order to maximize the ```operations / instruction``` we need to utilize our vector registers. Each 256 Bit vector registers can hold 8 32 Bit single precision floating point numbers. The ```_mm256_fmadd_ps``` intrinsic instruction operates on these registers and executes 8 fused multiply-add operations at once. This factor is 8/1.

Maximizing the ```instructions / cycle``` factor means to maximize the instruction throughput of the CPU. On the micro-architectures given above (see [4]) we have two execution units that can execute the ```_mm256_fmadd_ps``` instruction. Therefore, in case of two independent  ```_mm256_fmadd_ps``` instruction, this factor is 2/1.

For the last factor ```cylces / second``` we can use the max turbo-boost frequency of the specific processor.

E.g. for the Intel Core i7-7500U this yield theoretical Peak FLOPS of ```2 x 8 x 2 x 3.5GHz = 112.0 GFLOPS ``` per core by using the ```_mm256_fmadd_ps``` instruction and presumingly using both execution units.


### Practically maximizing FLOPS

Starting simple, consider the following code snippet. It performs a single ```_mm256_fmadd_ps``` operation for a number of iterations:

```c++
float run_kernel(size_t iterations)
{
	...
	do{
		r0 = _mm256_fmadd_ps(mul0, mul1, r0); // fused multiple add operation
	}while (--iterations);
	...
 }

```
While executing it, we measure 13.6 GFLOPS. Unfortunately, we have a series of dependent ```_mm256_fmadd_ps``` operations and therefore we utilize only one of our two execution units.

In order to use our second execution unit we can add a second, independent ```_mm256_fmadd_ps``` instruction:

```c++
float run_kernel(size_t iterations)
{
   		...
        do{
            r0 = _mm256_fmadd_ps(mul0, mul1, r0); // operation 1
            r1 = _mm256_fmadd_ps(mul0, mul1, r1); // operation 2

        }while (--iterations);
        ...
}
```
For this we measure 27.2GFLOPS. We doubled the number of FLOPS by using both of our execution units.

So far we maximized the instruction throughput. In order to maximize our experimental FLOPS, we need consider one more factor: Instruction latency.

According to [3], the latency of an instruction is the delay that the instruction generates in a dependency chain. The measurement unit is clock cycles.

The latency for the ```_mm256_fmadd_ps``` instruction vary, depending on your microarchitecture. According to [3], the Kaby Lake generation has a throughput of 2 and a latency of 4 for the  ```_mm256_fmadd_ps``` instruction.


Analyzing our previous result, we started two operations simultaniously but had to wait for 4 cycles until the operations completed.
In order to maximize our experimental PEAK Flops, we have to start one independent instructions per cycle for each execution unit. This means we need 4 x 2 = 8 independent operations:

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

Benchmarking this snipped yields 110.6 GFLOPS, or 98.7% of our precaclulated theoretical peak FLOPS.

In the following graph you can see, how it scales and peaks:

![Alt text](peak_flops.png?raw=true)


### Additional Information
- The Intel compiler compiler does not work.
- Thread support is comming.
- Benchmarking the code with Intel VTune Amplifier yields similar results.
- Check the assembly output if the results seem unreasonable.

## Resources:
##### [1] Theoretical Peak FLOPS per instruction set on modern Intel CPUs - http://www.dolbeau.name/dolbeau/publications/peak.pdf
##### [2] How to get peak FLOPS - https://www.eidos.ic.i.u-tokyo.ac.jp/~tau/lecture/parallel_distributed/2018/slides/pdf/peak_cpu.pdf
##### [3] Agner Instruction Tables - https://www.agner.org/optimize/instruction_tables.pdf
##### [4] Agner Microarchitecture -  https://www.agner.org/optimize/microarchitecture.pdf 


