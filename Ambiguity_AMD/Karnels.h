#ifndef _AMBIGUITY_KERNELS_H_
#define _AMBIGUITY_KERNELS_H_

#include <hip/hip_complex.h> 
#include <hip/hip_runtime.h>

// U¿yj extern dla wszystkich kerneli
extern "C" __global__ void ShiftHIP(hipFloatComplex* __restrict__ In, hipFloatComplex* __restrict__ Out, size_t shift, size_t cudaN);


extern "C" __global__ void ConvertAndScaleHIP(const float* __restrict__ raw_in, hipFloatComplex* __restrict__ complex_out, size_t Nth, float scale);


// Zmieniono z 4 na 5 argumentów.
extern "C" __global__ void CorelateShiftHIP(hipFloatComplex* __restrict__ BufX,
	hipFloatComplex* __restrict__ BufY,
	hipFloatComplex* __restrict__ BufW,
	size_t cudaCol, size_t cudaN);

extern "C" __global__ void MagnitudeHIP_optimized(hipFloatComplex* Inp, float* Out, int cuda_row, int cuda_col, int cuda_shift, int cudaN);





#endif // _AMBIGUITY_KERNELS_H_