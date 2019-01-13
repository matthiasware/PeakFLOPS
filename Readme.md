# Theoretical Peak FLOPS

Code for reaching the theoretical Peak FLOPS for the AVX2 instruction set on the following architectures:
Haswell - Broadwell - Skylake - Kaby Lake - Coffee Lake -Whiskey Lake - Amber Lake.

Calculating the theoretical peak FLOPS per core according to [1] can be done via:
$$\Big[\frac{flops}{second}\Big]_{core} = \frac{flops}{operation} \times \frac{operations}{instruction} \times \frac{instructions}{cycle} \times \frac{cycles}{second}$$

#### Example - Intel Core i7-7500U
For the Intel Core i7-7500U Processor [4] , using the AVX2 instruction set:
$\frac{flops}{operation}$: By using a fused multiply add, this is 2
$\frac{operations}{instruction}$: 256Bit registers, SP 32, 8 operations
$\frac{instructions}{cycle}$: This refers to the instruction throuput and is 2 (see [3])
$\frac{cycles}{second}$:  Under turbo boost: 3.5GHz

$$= 2 \times 8 \times 2 \times 3.5GHz = 112.0 \; GFLOPS $$
Additionlly we have to consider instruction latency [2]. From [3], we know that the fused multiply add operation has a latency of 4. Therefore with a throughput of 2 and a latency of 4 cylces for the fused multiply add, we should get peak performance by having 2 *4 = 8 independent operations.

```c++
int main()
```


### Validation
Since compilers are smart and try to optimize, it is highly advised to check the assembly code and use a profiler like Intel VTune Amplifier to verify that the operations are performed.

## Resources:
##### [1] Theoretical Peak FLOPS per instruction set on modern Intel CPUs - http://www.dolbeau.name/dolbeau/publications/peak.pdf
##### [2] How to get peak FLOPS - https://www.eidos.ic.i.u-tokyo.ac.jp/~tau/lecture/parallel_distributed/2018/slides/pdf/peak_cpu.pdf
##### [3] Agner Instruction Tables - https://www.agner.org/optimize/instruction_tables.pdf
##### [4] Agner Microarchitecture -  https://www.agner.org/optimize/microarchitecture.pdf 


