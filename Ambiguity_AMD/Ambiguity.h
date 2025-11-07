////////////////////////////////////////////////////////////////////////////////
//
//                          Super Fast Ambiguity Library
// Version 3.0
// Ambiguity.h : Defines the external interface and error codes for the DLL.
//
// Created by Daniel M. Kaminski (CUDA) / Converted to HIP by Gemini
//
// Year 2022/2025
//
////////////////////////////////////////////////////////////////////////////////


#ifndef _DLL_H_
#define _DLL_H_
#pragma once

// Zmiana na wersjê HIP
#define Version 3.0 // Zaktualizowana wersja po konwersji

#define PARALLEL					// Macro for OpenMP parallelism on the Host CPU
#define CALCULATE_SQRT_ON_CPU		// It is slightly faster on i5 11400 CPU
#define PRECOMPUTE					// Instead of calculating in the hot loop on the GPU, we use tables with precomputed values
#define PINED                       //Use pined memory
#define DEFOULT_CUDA_DEVICE         0 // Zachowano nazwê makra, ale odnosi siê do urz¹dzenia HIP/ROCm
#define ONE                         1
#define MAX_SHIFT                   400
#define CONSTANT_AMPLIFICATION      2000.0f
// This scaling is necessary to shift the dot and fit the number within the float size
// of 3.402823466 E + 38 after the FFT.
#define SCALE                       1E-11f

// Warning suppression:
#pragma warning(disable:4273) // This is due to function mismatch between header and implementation; generally safe to ignore.
#pragma warning(disable:4267) // Conversion warnings for size_t are suppressed, as these values are expected to be small.

 
// DLL Export/Import mechanism setup
// AMBIGUITYCUDA_EXPORTS must be defined in the DLL project settings when building.
#if defined(AMBIGUITYCUDA_EXPORTS) // Ensures __declspec(dllexport) when building the DLL
# define DLLIMPORT __declspec (dllexport)
#else /* Not BUILDING_DLL */
# define DLLIMPORT __declspec (dllimport)
#endif /* Not BUILDING_DLL */


 
#include <hipfft.h> 

// Nowy nag³ówek dla kerneli (funkcji device)
 

#define MAX_DEVICE_NAME 256

extern "C"
{
	////////////////////////////////////////////////////////////////////////////
	// FUNCTION: Initialize
	//
	// DESCRIPTION: Initializes the HIP device, allocates all necessary GPU
	//              memory buffers, creates hipFFT plans, and configures streams.
	//
	// INPUT:
	//   BufferSize:  Total size of the input buffer (must be divided by 2).
	//   col:         Number of columns for the ambiguity map (Doppler dimension).
	//   row:         Number of rows for the ambiguity map (Time dimension).
	//   doppler_shift: Doppler shift parameter (currently unused).
	//   name:        Pointer to a short array to receive the GPU device name.
	//
	// OUTPUT:
	//   int: Returns CUDA_OK (0) on success, or a specific negative error code.
	////////////////////////////////////////////////////////////////////////////
	int DLLIMPORT Initialize(unsigned int BufferSize, unsigned int col, unsigned int row, float doppler_shift, short* name);

	////////////////////////////////////////////////////////////////////////////
	// FUNCTION: Run
	//
	// DESCRIPTION: Main processing function. Executes the FFT, correlation,
	//              inverse FFT, magnitude calculation, and post-processing.
	//
	// INPUT:
	//   Data_In0:      Raw float array (Real/Imag interleaved) for the first dongle.
	//   Data_In1:      Raw float array (Real/Imag interleaved) for the second dongle.
	//   Data_Out:      Output buffer for the final ambiguity map (Host).
	//   amplification: Global amplification factor for post-processing.
	//   doppler_zoom:  Scaling factor for the Doppler axis.
	//   time_shift:    Shift applied to the Time axis (used for magnitude cropping).
	//   two_dongles:   Flag: true if two dongles are used (cross-correlation mode).
	//   scale_type:    Type of scaling/leveling post-processing (0, 1, or 2).
	//   remove_symetrics: Flag to remove symmetry artifacts.
	//
	// OUTPUT:
	//   int: Returns CUDA_OK (0) on success, or a specific negative error code.
	////////////////////////////////////////////////////////////////////////////
	int DLLIMPORT Run(float* Data_In0, float* Data_In1, float* Data_Out, float amplification, float doppler_zoom, int time_shift, bool two_dongles, short scale_type, bool remove_symetrics);

	////////////////////////////////////////////////////////////////////////////
	// FUNCTION: Release
	//
	// DESCRIPTION: Destroys all hipFFT plans, frees all GPU memory (hipFree),
	//              destroys the HIP stream, and performs a device reset.
	//
	// INPUT: None.
	//
	// OUTPUT:
	//   int: Returns CUDA_OK (0) on success, or a specific negative error code.
	////////////////////////////////////////////////////////////////////////////
	int DLLIMPORT  Release();


#ifdef PINED
	////////////////////////////////////////////////////////////////////////////
	// FUNCTION: AllocHostMemory
	// Description: Allocates two page-locked (Pinned) Host memory buffers.
	//
	// INPUT:
	// bufferSize: Total number of float elements required for a single buffer.
	//
	// OUTPUT:
	// hostPtr0:   Pointer to the first allocated Pinned Memory buffer (mapped to 'out IntPtr' in C#).
	// hostPtr1:   Pointer to the second allocated Pinned Memory buffer (mapped to 'out IntPtr' in C#).
	////////////////////////////////////////////////////////////////////////////
	int DLLIMPORT AllocHostMemory(unsigned bufferSize, float** hostPtr0, float** hostPtr1);

	////////////////////////////////////////////////////////////////////////////
	// FUNCTION: FreeHostMemory
	// Description: Frees the page-locked (Pinned) Host memory buffers.
	//
	// INPUT/OUTPUT:
	// hostPtr0:   Pointer to the first buffer (mapped to 'ref IntPtr' in C#). Reset to nullptr after free.
	// hostPtr1:   Pointer to the second buffer (mapped to 'ref IntPtr' in C#). Reset to nullptr after free.
	////////////////////////////////////////////////////////////////////////////
	int DLLIMPORT FreeHostMemory(float** hostPtr0, float** hostPtr1);
#endif
}


// Internal Host Function Prototypes (zmienione nazwy funkcji CUDA na HIP)
int   FFT_forward();
int   FFT_backward();
int   CalcCorrelateShift();
int   CalcShift(size_t rotation_shift);
int   Magnitude(int shift);
int   StreamSynchronise();
int   Synchronise();
float FindMin(float x, float y);
int   ConvertAndScale(float* Cuda_RawIn, hipFloatComplex* Cuda_bufOut);
int   SetupConstantMemory(float zoom, size_t maxCol, size_t Nth);


// CUDA Kernel and Device Function Prototypes (usuniête, przeniesione do Ambiguity_Kernels.h)
// U¿ywamy hipLaunchKernelGGL do uruchamiania kerneli.
  __global__ void ShiftHIP(hipFloatComplex* __restrict__ In, hipFloatComplex* __restrict__ Out, size_t shift, size_t cudaN);


  __global__ void ConvertAndScaleHIP(const float* __restrict__ raw_in, hipFloatComplex* __restrict__ complex_out, size_t Nth, float scale);


// Zmieniono z 4 na 5 argumentów.
  __global__ void CorelateShiftHIP(hipFloatComplex* __restrict__ BufX,
	hipFloatComplex* __restrict__ BufY,
	hipFloatComplex* __restrict__ BufW,
	size_t cudaCol, size_t cudaN);

  __global__ void MagnitudeHIP_optimized(hipFloatComplex* Inp, float* Out, int cuda_row, int cuda_col, int cuda_shift, int cudaN);


#endif /* _DLL_H_ */


//==============================================================================
// ERROR CODES
//==============================================================================
// Zachowano oryginalne kody b³êdów
#define CUDA_OK                                  0
#define CUDA_RUNNING                            -1  // General runtime error, potentially wrong column number
#define CUDA_COLUMN_IN_WRONG_RANGE              -2
#define CUDA_TOO_MUCH_REQUESTED_MEMORY          -3
#define CUDA_MEMORY_COPY_ERROR                  -10
#define CUDA_FFT_ERROR                          -11 // General hipFFT error
#define CUDA_STREAM_CREATE_ERROR                -12
#define CUDA_STREAM_SET_ERROR                   -13
#define CUDA_FFT_PLAN1D_CREATE_ERROR            -14
#define CUDA_FFT_CREATE_ERROR                   -15
#define CUDA_FFT_EXECUTE_ERROR                  -16


// ERRORS are additionally decorated with CUDA runtime/driver errors (-XXXX-cudaError)
#define CUDA_DEVICE_SYNCHRONISATION_ERROR       -1000
#define CUDA_SHIFT_CALCULATION_ERROR            -2000
#define CUDA_SHIFT_CORELATE_ERROR               -3000
#define CUDA_MAGNITUDE_ERROR                    -4000
#define CUDA_CORELATE_ERROR                     -5000
#define CUDA_SHIFT_ERROR                        -6000
#define CUDA_DEVICE_RESET_ERROR                 -7000
#define CUDA_STREAM_DESTROY_ERROR               -8000
#define CUDA_CUFFT_DESTROY_ERROR                -9000
#define CUDA_FREE_ERROR                         -10000
#define CUDA_MALLOC_ERROR                       -11000
#define CUDA_MALLOC_ERROR2                      -11200
#define CUDA_MALLOC_FREE_ERROR2                 -11300
#define CUDA_SET_DEVICE_ERROR                   -12000
#define CUDA_GET_DEVICE_ERROR                   -13000
#define CUDA_STREAM_SYNCHRONISATION_ERROR       -14000
#define CUDA_CONVERT_ERROR                      -15000
#define CUDA_MEMCPY_ERROR                       -16000

/*
* hipFFT Error List (values are added to base errors, e.g., -14 - CUFFT_PLAN1D_CREATE_ERROR)
  HIPFFT_SUCCESS             = 0x0,
  HIPFFT_INVALID_PLAN        = 0x1,
  HIPFFT_ALLOC_FAILED        = 0x2,
  HIPFFT_INVALID_TYPE        = 0x3,
  HIPFFT_INVALID_VALUE       = 0x4,
  HIPFFT_INTERNAL_ERROR      = 0x5,
  HIPFFT_EXEC_FAILED         = 0x6,
  HIPFFT_SETUP_FAILED        = 0x7,
  HIPFFT_INVALID_SIZE        = 0x8,
  HIPFFT_UNALIGNED_DATA      = 0x9,
  HIPFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
  HIPFFT_INVALID_DEVICE      = 0xB,
  HIPFFT_PARSE_ERROR         = 0xC,
  HIPFFT_NO_WORKSPACE        = 0xD,
  HIPFFT_NOT_IMPLEMENTED     = 0xE,
  HIPFFT_LICENSE_ERROR       = 0x0F,
  HIPFFT_NOT_SUPPORTED       = 0x10
*/