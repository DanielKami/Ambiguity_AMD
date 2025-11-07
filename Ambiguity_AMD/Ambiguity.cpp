////////////////////////////////////////////////////////////////////////////////
//
//                          Super Fast Ambiguity Library
// Version 3.0
// Ambiguity.cpp : Defines the exported functions for the DLL.
//
// Created by Daniel M. Kaminski (CUDA) / Converted to HIP by Gemini
//
// Year 2022/2025
//
////////////////////////////////////////////////////////////////////////////////

// HIP (ROCm) toolkit is used
// https://rocm.github.io/

// General Information:
//
// Input: Data are float strings from RTL one or two dongles.
// Output: Data in map format (x,y) containing Doppler shift (X) and time delay (Y).
//
// Warning:
//   ERROR codes are defined in the header file!
//   The settings are generally safe for modern GPUs, as this library operates mostly on memory.
//   Maximum estimated memory usage (for Nth=1024, Col=512, Row=1024, sizeof(hipFloatComplex)=8 bytes) is below 6GB.



#include "stdafx.h"
#include <stdio.h>
#include <omp.h> // Wymagane dla PARALLEL

// Zmiana nag³ówków CUDA na HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipfft/hipfft.h> 

#include "ambiguity.h"
 


#ifdef PRECOMPUTE
// Array for pre-calculated shifts for index 'j' (shared_zoom * n)
__constant__  size_t constant_ShiftOffsets[1024];

// Array for pre-calculated offsets for index 'output_index' (n * N)
__constant__ size_t constant_OutputOffsets[1024];
#endif

__constant__ float constant_CudaZoom;

// Globals
bool openCL_Initiated = false;

// CUDA pointers (6GB space available) -> Zmienione na HIP
// U¿ywamy hipFloatComplex zamiast float2
hipFloatComplex* Cuda_bufX = nullptr;
hipFloatComplex* Cuda_bufY = nullptr;
hipFloatComplex* Cuda_bufW = nullptr;

// Zmiana z __constant__ na __constant__ w HIP
// Zmiana z __constant__ na __constant__ w HIP


int numBlocks;
int blockSize;

// U¿ywamy hipFloatComplex zamiast float2
hipFloatComplex* tmpData = nullptr;
float* Cuda_ColRow = nullptr;
float* Cuda_RawIn0 = nullptr;
float* Cuda_RawIn1 = nullptr;

// Zmiana z cudaStream_t na hipStream_t
hipStream_t stream = nullptr;

// Zmiana z cufftHandle na hipfftHandle
hipfftHandle plan1;
hipfftHandle plan2;

size_t Nth;
size_t N_Float2InBites;
size_t N_Float_Bites;

int col, row;
int ColRow;

float doppler_zoom_old;
int col_old;
int Nth_old;

int HalfCol;
float doppler_shift_zoom_f;

size_t sizeR;
size_t sizeN;


#ifdef PRECOMPUTE
////////////////////////////////////////////////////////////////////////////////
// FUNCTION: SetupConstantMemory
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int SetupConstantMemory(float zoom, size_t maxCol, size_t Nth) {
	if (maxCol > 1024) return CUDA_RUNNING; // Constant array limitation

	// Allocation of temporary buffers on CPU
	size_t* h_ShiftOffsets = new size_t[maxCol];
	size_t* h_OutputOffsets = new size_t[maxCol];

	// 1. Calculation of values on CPU
	for (size_t n = 0; n < maxCol; ++n) {
		// Pre-calculation of 'shift' for index j: i + shift
		// We use the roundf function because casting float to size_t truncates,
		// and we want the nearest integer shift value.
		h_ShiftOffsets[n] = static_cast<size_t>(roundf(zoom * (float)n));

		// Pre-calculation of 'offset' for output index: i + offset
		h_OutputOffsets[n] = n * Nth;
	}

	// 2. Kopiowanie danych do pamiêci sta³ej GPU (zmiana cudaMemcpyToSymbol na hipMemcpyToSymbol)
	hipError_t err1 = hipMemcpyToSymbol(HIP_SYMBOL(constant_ShiftOffsets), h_ShiftOffsets, maxCol * sizeof(size_t), 0, hipMemcpyHostToDevice);
	hipError_t err2 = hipMemcpyToSymbol(HIP_SYMBOL(constant_OutputOffsets), h_OutputOffsets, maxCol * sizeof(size_t), 0, hipMemcpyHostToDevice);

	delete[] h_ShiftOffsets;
	delete[] h_OutputOffsets;

	if (err1 != hipSuccess || err2 != hipSuccess) {
		return CUDA_MEMCPY_ERROR;
	}
	return CUDA_OK; // Success
}
#endif


#ifdef PINED

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: AllocHostMemory
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int   AllocHostMemory(unsigned bufferSize, float** hostPtr0, float** hostPtr1)
{
	// Initialize output pointers to nullptr for safety and clean error handling.
	*hostPtr0 = nullptr;
	*hostPtr1 = nullptr;

	size_t sizeBytes = bufferSize * sizeof(float);
	hipError_t err = hipSuccess;

	// 1. Allocation for the first buffer (Data_In0)
	// Zmiana cudaHostAlloc na hipHostAlloc
	err = hipHostAlloc((void**)hostPtr0, sizeBytes, hipHostMallocMapped);

	// Check for first allocation error
	if (err != hipSuccess)
	{
		return CUDA_MALLOC_ERROR - err;
	}

	// 2. Allocation for the second buffer (Data_In1)
	err = hipHostAlloc((void**)hostPtr1, sizeBytes, hipHostMallocMapped);

	// Check for second allocation error
	if (err != hipSuccess)
	{
		// CRITICAL STEP: If the second allocation fails, free the first one to prevent a memory leak.
		// Zmiana cudaFreeHost na hipHostFree
		hipHostFree(*hostPtr0);

		// Reset the pointer to nullptr after freeing for safety.
		*hostPtr0 = nullptr;

		return CUDA_MALLOC_ERROR2 - err;
	}

	return CUDA_OK;
}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: FreeHostMemory
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int   FreeHostMemory(float** hostPtr0, float** hostPtr1)
{
	hipError_t err = hipSuccess;

	// 1. Free the first buffer
	// Check if the pointer is not nullptr (prevents unnecessary call if already reset)
	if (*hostPtr0 != nullptr)
	{
		// Zmiana cudaFreeHost na hipHostFree
		err = hipHostFree(*hostPtr0);
		*hostPtr0 = nullptr; // KEY: Reset the original pointer to prevent double-free
	}
	// Check error only if the pointer was being freed
	if (err != hipSuccess) return CUDA_MALLOC_FREE_ERROR2;

	// 2. Free the second buffer
	if (*hostPtr1 != nullptr)
	{
		err = hipHostFree(*hostPtr1);
		*hostPtr1 = nullptr; // KEY: Reset the original pointer to prevent double-free
	}
	// Check error only if the pointer was being freed
	if (err != hipSuccess) return CUDA_MALLOC_FREE_ERROR2;

	return CUDA_OK;
}
#endif // PINED
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: Initialize
//
// ... [Cia³o funkcji] ...
////////////////////////////////////////////////////////////////////////////////
int Initialize(unsigned int BufferSize, unsigned int col_, unsigned int row_, float doopler_shift_, short* name)
{
	hipError_t hipStatus;

	// It must be divided by 2 because of the main program logic
	Nth = static_cast<size_t>(BufferSize) / 2;    // size is smaller than 32 bits, so it is OK
	HalfCol = col_ / 2;
	N_Float_Bites = sizeof(float) * Nth * 2;

	// Calculate optimal block size (for kernels without shared memory)
	int minGridSize;
	// POPRAWKA: Rzutowanie wskaŸnika do kernela na **const void***, aby unikn¹æ b³êdu dwuznacznoœci
	hipStatus = hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (const void*)ConvertAndScaleHIP, 0, Nth);

	// Konwersja z cudaSuccess na hipSuccess
	if (hipStatus != hipSuccess) {
		// Hip mo¿e nie wspieraæ tej funkcji dla kernela, wiêc u¿ywamy domyœlnego
		blockSize = 256;
	}

	// Defensive check against zero (a fix for a previous runtime error)
	if (blockSize == 0) blockSize = 256;

	// Calculate number of blocks
	numBlocks = (Nth + blockSize - 1) / blockSize;

	// Protect against wrong column number
	if (col_ < 16 || col_ > 1024) return CUDA_RUNNING;

	col = col_;
	row = row_;

	ColRow = col * row;
	// N_Float2InBites - hipFloatComplex ma ten sam rozmiar co float2
	N_Float2InBites = sizeof(hipFloatComplex) * Nth * ONE;

	// Machine (PC) pointers
	tmpData = new hipFloatComplex[Nth];
	//cudaMallocHost((void**)&tmpData, N_InBites);

	int count = 0;
	// Zmiana cudaGetDeviceCount na hipGetDeviceCount
	hipGetDeviceCount(&count);

	// Zmiana cudaDeviceProp na hipDeviceProp_t
	hipDeviceProp_t prop;
	// Zmiana cudaGetDeviceProperties na hipGetDeviceProperties
	hipStatus = hipGetDeviceProperties(&prop, DEFOULT_CUDA_DEVICE);
	// Zmiana cudaSuccess na hipSuccess
	if (hipStatus != hipSuccess)
	{
		return CUDA_GET_DEVICE_ERROR;
	}

	// Copy the device name to short pointer. (Ugly way)
	for (int i = 0; i < MAX_DEVICE_NAME; i++)
		name[i] = prop.name[i];

	// Choose which GPU to run on. 0 is default.
	// Zmiana cudaSetDevice na hipSetDevice
	hipStatus = hipSetDevice(0);
	if (hipStatus != hipSuccess) {
		return CUDA_SET_DEVICE_ERROR - hipStatus;
	}

	// Zmiana cufftCreate na hipfftCreate
	if (hipfftCreate(&plan1) != HIPFFT_SUCCESS) {
		return CUDA_FFT_CREATE_ERROR;
	}

	// Plan for single column FFT (used for reference data)
	// Zmiana cufftPlan1d na hipfftPlan1d (CUFFT_C2C -> HIPFFT_C2C)
	if (hipfftPlan1d(&plan1, (int)Nth, HIPFFT_C2C, ONE) != HIPFFT_SUCCESS) {
		return CUDA_FFT_PLAN1D_CREATE_ERROR;
	}

	// Parameters
	int rank = ONE;                   // 1D transformation
	int n[] = { (int)Nth };           // Transform length
	int batch = (int)col;             // Number of transforms in the batch (batched)

	// Stride indicators:
	int istride = ONE;                // Continuous elements inside the transformed vector
	int idist = (int)Nth;             // Distance in elements from the beginning of one column to the next

	// If you use the same buffer Cuda_bufW as input and output (in-place):
	int* inembed = nullptr;
	int* onembed = nullptr;
	int ostride = istride;
	int odist = idist;

	// Zmiana cufftPlanMany na hipfftPlanMany (CUFFT_C2C -> HIPFFT_C2C)
	hipfftPlanMany(&plan2, rank, n, inembed, istride, idist, onembed, ostride, odist, HIPFFT_C2C, batch);

	//*********************************************************************
	// Memory allocation
	//*********************************************************************

	int su = 0; // returned su is negative

	// Buffer X (Input/Working buffer for FFT of data 1 or 2)
	// Zmiana cudaMalloc na hipMalloc (dla hipFloatComplex)
	hipMalloc(reinterpret_cast<void**>(&Cuda_bufX), N_Float2InBites);
	if ((su = hipGetLastError()) != hipSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	// Buffer Y (Reference data buffer)
	hipMalloc(reinterpret_cast<void**>(&Cuda_bufY), N_Float2InBites);
	if ((su = hipGetLastError()) != hipSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	// Buffer W (Output buffer for Corelation/Ambiguity map)
	hipMalloc(reinterpret_cast<void**>(&Cuda_bufW), N_Float2InBites * col);
	if ((su = hipGetLastError()) != hipSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	// Buffer for final result (Magnitude)
	hipMalloc(reinterpret_cast<void**>(&Cuda_ColRow), sizeof(float) * (ColRow + MAX_SHIFT));
	if ((su = hipGetLastError()) != hipSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}


	// Nth is the number of hipFloatComplex elements, so 2*Nth is the number of floats.
	size_t N_Float_Bites = sizeof(float) * Nth * 2;

	// Buffer for raw input 0
	hipMalloc(reinterpret_cast<void**>(&Cuda_RawIn0), N_Float_Bites);
	if ((su = hipGetLastError()) != hipSuccess) { return CUDA_MALLOC_ERROR - su; }

	// Buffer for raw input 1 (needed only in two-dongle mode)
	hipMalloc(reinterpret_cast<void**>(&Cuda_RawIn1), N_Float_Bites);
	if ((su = hipGetLastError()) != hipSuccess) { return CUDA_MALLOC_ERROR - su; }

	// Create stream for non-blocking operations
	// Zmiana cudaStreamCreateWithFlags na hipStreamCreateWithFlags (cudaStreamNonBlocking -> hipStreamNonBlocking)
	if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
		return CUDA_STREAM_CREATE_ERROR;
	}

	// Set the stream for both plans
	// Zmiana cufftSetStream na hipfftSetStream
	if (hipfftSetStream(plan1, stream) != HIPFFT_SUCCESS) {
		return CUDA_STREAM_SET_ERROR;
	}

	if (hipfftSetStream(plan2, stream) != HIPFFT_SUCCESS) {
		return CUDA_STREAM_SET_ERROR;
	}

	openCL_Initiated = true;
	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: Run
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
// Data_In is in a format Data_In[2*n] - Real values, Data_In[2*n+1] - Imaginary values
int Run(float* Data_In0, float* Data_In1, float* Data_Out, float amplification, float doppler_zoom, int time_shift, bool two_dongles, short scale_type, bool remove_symetrics)
{
	int err;
	float maxval = 1E-6f;


	if (!openCL_Initiated)
		return CUDA_RUNNING;

	if (time_shift > MAX_SHIFT)
		time_shift = MAX_SHIFT;

	if (doppler_zoom_old != doppler_zoom || col_old != col || Nth_old != Nth)
	{
		doppler_shift_zoom_f = 1.0f * Nth / doppler_zoom;
		float doppler_zoom_Col = doppler_shift_zoom_f / col;

#ifdef PRECOMPUTE
		err = SetupConstantMemory(doppler_zoom_Col, col, Nth);
		if (err < CUDA_OK)
			return err;
#else
		// Set constant memory for the kernel
		// Zmiana cudaMemcpyToSymbol na hipMemcpyToSymbol
		hipError_t status = hipMemcpyToSymbol(HIP_SYMBOL(constant_CudaZoom), &Doppler_zoom_Col, sizeof(float), 0, hipMemcpyHostToDevice);
		if (status != hipSuccess) return CUDA_SET_DEVICE_ERROR - status;
#endif

		doppler_shift_zoom_f /= 2;// it is used only once in the code, just divide by two for correct use in next function

		doppler_zoom_old = doppler_zoom;
		col_old = col;
	}


	// 1. Process First data0 (Reference)
	// Zmiana cudaMemcpyAsync na hipMemcpyAsync (cudaMemcpyHostToDevice -> hipMemcpyHostToDevice)
	if (hipMemcpyAsync(Cuda_RawIn0, Data_In0, N_Float_Bites, hipMemcpyHostToDevice, stream) != hipSuccess)
	{
		return CUDA_MEMORY_COPY_ERROR;
	}

	// Conversion and Scaling on GPU (Kernel)
	err = ConvertAndScale(Cuda_RawIn0, Cuda_bufX);
	if (err < CUDA_OK)
		return err;

	// Fourier Transform of reference data (shift 0 deg)
	// Input/Output: Cuda_bufX
	err = FFT_forward();
	if (err < CUDA_OK)
		return err;

	// Rotate Cuda_bufX after fft to the Col/2 (screen middle) -> Cuda_bufY (reference)
	err = CalcShift((size_t)(doppler_shift_zoom_f));
	if (err < CUDA_OK)
		return err;

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// 2. Process Second Dongle (or use the first one again)
	///////////////////////////////////////////////////////////////////////////////////////////////////
	if (two_dongles)
	{
		if (hipMemcpyAsync(Cuda_RawIn1, Data_In1, N_Float_Bites, hipMemcpyHostToDevice, stream) != hipSuccess)
		{
			return CUDA_MEMORY_COPY_ERROR;
		}

		err = ConvertAndScale(Cuda_RawIn1, Cuda_bufX);
		if (err < CUDA_OK)
			return err;

		// FFT of the second dongle in buffer Cuda_bufX
		err = FFT_forward();
		if (err < CUDA_OK)
			return err;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 3. Ambiguity Map Calculation
	// Input:
	//    Cuda_bufY - FFT reference data, shifted to half (Doppler center)
	//    Cuda_bufX - FFT basic data  
	// Output: Cuda_bufW
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Calculate ambiguity
	// The the hard part of calculations (shift every column)
	// Loops are moved to the threads inside GPU
	err = CalcCorrelateShift();
	if (err < CUDA_OK)
		return err;

	// Inverse FFT (Batch operation)
	// Input/Output: Cuda_bufW
	// Computed in parallel for BATCH = Col, x = Nth (very expensive)
	err = FFT_backward();
	if (err < CUDA_OK)
		return err;

	// Magnitude Calculation
	// Input: Cuda_bufW
	// Output: Cuda_ColRow
	err = Magnitude(time_shift * 2);
	if (err < CUDA_OK)
		return err;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 4. Copy to Host (expensive operation)
	// Copy all results in one shot to save time.
	// Zmiana cudaMemcpyAsync na hipMemcpyAsync (cudaMemcpyDeviceToHost -> hipMemcpyDeviceToHost)
	if (hipMemcpyAsync(Data_Out, Cuda_ColRow, sizeof(float) * (static_cast<unsigned long long>(ColRow) + time_shift), hipMemcpyDeviceToHost, stream) != HIPFFT_SUCCESS)
		return CUDA_MEMORY_COPY_ERROR;

	err = StreamSynchronise();
	if (err < CUDA_OK)
		return err;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//                        Signal Map Post-processing (Host CPU)
	//
	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Find max
#ifdef PARALLEL
#pragma omp parallel for reduction(max: maxval)
#endif
#ifdef CALCULATE_SQRT_ON_CPU
	for (int n = 0; n < ColRow; n += 4) {
		// Przetwarzaj 4 elementy na raz
		for (int k = 0; k < 4 && (n + k) < ColRow; k++) {
			Data_Out[n + k] = sqrtf(Data_Out[n + k]);
			maxval = fmaxf(maxval, Data_Out[n + k]);
		}
	}
#else
	for (int n = 0; n < ColRow; n += 4) {
		// Przetwarzaj 4 elementy na raz
		for (int k = 0; k < 4 && (n + k) < ColRow; k++) {
			maxval = fmaxf(maxval, Data_Out[n + k]);
		}
	}
#endif
	// Auto signal leveling
	double totalMAX = 0;
	maxval = 1.0f / maxval;

	// Normalise to max value
	for (int n = 0; n < ColRow; ++n)
	{
		totalMAX += (Data_Out[n] *= maxval);
	}

	// Set safe limits for TotalMAX
	if (totalMAX <= 10)
		totalMAX = 10;
	if (totalMAX > 1e30)
		totalMAX = 1e30;

	float TotalMAXf = (float)(CONSTANT_AMPLIFICATION * amplification / totalMAX);

	// Set the correct level for map based on scale_type
	if (scale_type == 0)
	{
		for (int n = 0; n < ColRow; ++n)
		{
			Data_Out[n] *= TotalMAXf;
		}
	}
	else if (scale_type == 1)
	{
		for (int n = 0; n < ColRow; n++)
		{
			// Data_Out[n] is already sqrt(Magnitude)
			Data_Out[n] *= Data_Out[n] * TotalMAXf * 10;
		}
	}
	else if (scale_type == 2)
	{
		for (int n = 0; n < ColRow; n++)
		{
			// Data_Out[n] is already sqrt(Magnitude)
			Data_Out[n] *= Data_Out[n] * Data_Out[n] * TotalMAXf * 100;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//                              Remove Symetrics
	//
	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int c1, c2;
	int tmp1, tmp2;
	float min;

	// Mirror plane symmetry reduce intensity
	if (remove_symetrics)
	{
#ifdef PARALLEL
#pragma omp parallel for shared (Row)
#endif
		for (int C = 1; C < HalfCol + 1; C++)
		{
			c1 = (HalfCol + C) * row;
			c2 = (HalfCol - C) * row;

			for (int R = 0; R < row; ++R)
			{
				tmp1 = c1 + R;
				tmp2 = c2 + R;
				min = -FindMin(Data_Out[tmp1], Data_Out[tmp2]) * 0.5f;
				Data_Out[tmp1] += min;
				Data_Out[tmp2] += min;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Done, return corrected image
	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: FindMin
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
inline float FindMin(float x, float y)
{
	if (x < y)
		return (x);
	return (y);
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: StreamSynchronise
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int StreamSynchronise()
{
	int su;
	// Zmiana cudaStreamSynchronize na hipStreamSynchronize
	if ((su = hipStreamSynchronize(stream)) != hipSuccess)
	{
		return CUDA_STREAM_SYNCHRONISATION_ERROR - su;
	}
	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: Synchronise
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int Synchronise()
{
	int su;
	// Zmiana cudaDeviceSynchronize na hipDeviceSynchronize
	if ((su = hipDeviceSynchronize()) != hipSuccess)
	{
		return CUDA_DEVICE_SYNCHRONISATION_ERROR - su;
	}
	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: FFT_forward
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
// In/Out: Cuda_bufX
int FFT_forward()
{
	// Forward fft
	// Zmiana cufftExecC2C na hipfftExecC2C (CUFFT_FORWARD -> HIPFFT_FORWARD)
	if (hipfftExecC2C(plan1, Cuda_bufX, Cuda_bufX, HIPFFT_FORWARD) != HIPFFT_SUCCESS)
	{
		return CUDA_FFT_EXECUTE_ERROR;
	}

	// The same stream is used, no explicit synchronisation is needed here.
	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: FFT_bacward
//
// ... [Cia³o funkcji] ...
//
// NOTES:
//   W CuFFT u¿yto CUFFT_INVERSE. W hipFFT nale¿y u¿yæ **HIPFFT_BACKWARD**
////////////////////////////////////////////////////////////////////////////////

int FFT_backward()
{
	// Backward fft
	// POPRAWKA: Zmiana cufftExecC2C na hipfftExecC2C (CUFFT_INVERSE -> **HIPFFT_BACKWARD**)
	if (hipfftExecC2C(plan2, Cuda_bufW, Cuda_bufW, HIPFFT_BACKWARD) != HIPFFT_SUCCESS)
	{
		return CUDA_FFT_EXECUTE_ERROR;
	}

	// Synchronisation is crucial before magnitude calculation/copy.
	int err = Synchronise();
	if (err < CUDA_OK)
		return err;

	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: Release
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int Release()
{
	hipError_t hipStatus;

	if (!openCL_Initiated)
		return CUDA_OK; // If not initialized, return for safety

	/* Release HIP memory objects. */

	// Zmiana cudaFree na hipFree
	hipStatus = hipFree(Cuda_bufX);
	if (hipStatus != hipSuccess) {
		return CUDA_FREE_ERROR - hipStatus;
	}
	hipStatus = hipFree(Cuda_bufY);
	if (hipStatus != hipSuccess) {
		return CUDA_FREE_ERROR - hipStatus;
	}

	hipStatus = hipFree(Cuda_bufW);
	if (hipStatus != hipSuccess) {
		return CUDA_FREE_ERROR - hipStatus;
	}

	hipStatus = hipFree(Cuda_ColRow);
	if (hipStatus != hipSuccess) {
		return CUDA_FREE_ERROR - hipStatus;
	}

	hipStatus = hipFree(Cuda_RawIn0);
	if (hipStatus != hipSuccess) {
		return CUDA_FREE_ERROR - hipStatus;
	}

	hipStatus = hipFree(Cuda_RawIn1);
	if (hipStatus != hipSuccess) {
		return CUDA_FREE_ERROR - hipStatus;
	}

	int su;
	// Zmiana cufftDestroy na hipfftDestroy
	if ((su = hipfftDestroy(plan1)) != HIPFFT_SUCCESS) {
		return CUDA_CUFFT_DESTROY_ERROR - su;
	}

	if ((su = hipfftDestroy(plan2)) != HIPFFT_SUCCESS) {
		return CUDA_CUFFT_DESTROY_ERROR - su;
	}

	// Zmiana cudaStreamDestroy na hipStreamDestroy
	hipStatus = hipStreamDestroy(stream);
	if (hipStatus != hipSuccess) {
		return CUDA_STREAM_DESTROY_ERROR - hipStatus;
	}

	// cudaDeviceReset must be called before exiting for profiling tools to show complete traces.
	// Zmiana cudaDeviceReset na hipDeviceReset
	hipStatus = hipDeviceReset();
	if (hipStatus != hipSuccess) {
		return CUDA_DEVICE_RESET_ERROR - hipStatus;
	}

	// Machine pointers
	delete[] tmpData;

	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: ConvertAndScale
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int ConvertAndScale(float* Cuda_RawIn, hipFloatComplex* Cuda_bufOut)
{
	// Launch kernel (zmiana sk³adni uruchamiania kerneli na hipLaunchKernelGGL)
	hipLaunchKernelGGL(ConvertAndScaleHIP,
		dim3(numBlocks),
		dim3(blockSize),
		0,
		stream,
		Cuda_RawIn,
		Cuda_bufOut,
		Nth,
		SCALE
	);

	int err;
	// Zmiana cudaGetLastError na hipGetLastError
	if ((err = hipGetLastError()) != hipSuccess) {
		return CUDA_CONVERT_ERROR - err;
	}

	// Using streams means synchronization is not necessary here (it is handled by FFT_forward()).
	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: CalcShift
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int CalcShift(size_t rotation_shift)
{
	// Launch kernel (zmiana sk³adni uruchamiania kerneli na hipLaunchKernelGGL)
	hipLaunchKernelGGL(ShiftHIP,
		dim3(numBlocks),
		dim3(blockSize),
		0,
		stream,
		Cuda_bufX,
		Cuda_bufY,
		rotation_shift,
		Nth
	);

	int err;
	if ((err = hipGetLastError()) != hipSuccess) {
		return CUDA_SHIFT_ERROR - err;
	}

	//err = cudaStreamSynchronize(stream);
	//if (err < CUDA_OK)
	//	return err;

	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: CalcCorrelateShift
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
// Ambiguity.cu
int CalcCorrelateShift()
{
	// Rêczne ustawienie rozmiaru bloku optymalnego dla tego typu operacji
	// (czêsto 128, 256 lub 512). U¿ycie 256 jest bezpieczne.
	const int CORE_BLOCK_SIZE = 1024;

	// U¿ywamy sta³ego Nth (cudaN)
	size_t N = Nth;

	// 1. Obliczenie liczby bloków
	int local_numBlocks = (N + CORE_BLOCK_SIZE - 1) / CORE_BLOCK_SIZE;

	// 2. Obliczenie wymaganej pamiêci wspó³dzielonej
	// Jeden blok potrzebuje miejsca na CORE_BLOCK_SIZE elementów hipFloatComplex
	size_t sharedMemSize = CORE_BLOCK_SIZE * sizeof(hipFloatComplex);

	// Launch kernel (zmiana sk³adni uruchamiania kerneli na hipLaunchKernelGGL)
	hipLaunchKernelGGL(CorelateShiftHIP,
		dim3(local_numBlocks),
		dim3(CORE_BLOCK_SIZE),
		sharedMemSize,
		stream,
		Cuda_bufX,
		Cuda_bufY,
		Cuda_bufW,
		col,
		N
	);

	int err;
	if ((err = hipGetLastError()) != hipSuccess) {
		return CUDA_SHIFT_ERROR - err;
	}
	return CUDA_OK;
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: Magnitude
//
// ... [Cia³o funkcji bez zmian] ...
////////////////////////////////////////////////////////////////////////////////
int Magnitude(int shift)
{
	// Configure block and grid for 2D kernel launch
	const dim3 threads_per_block(16, 16);
	dim3 num_blocks((col + threads_per_block.x - 1) / threads_per_block.x,
		(row + threads_per_block.y - 1) / threads_per_block.y);

	// Launch kernel (zmiana sk³adni uruchamiania kerneli na hipLaunchKernelGGL)
	hipLaunchKernelGGL(MagnitudeHIP_optimized,
		num_blocks,
		threads_per_block,
		0,
		stream,
		Cuda_bufW,
		Cuda_ColRow,
		row,
		col,
		shift,
		Nth
	);

	int err;
	if ((err = hipGetLastError()) != hipSuccess) {
		return CUDA_SHIFT_ERROR - err;
	}
	return CUDA_OK;
}




///////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////



/*
*DEVICE FUNCTION : complex_conj_mult_opt
*
*DESCRIPTION : Performs the complex conjugate multiplication : W = Y * X *
*U¿ywa standardowej funkcji * *fmaf * *(Fused Multiply - Add).
*
*INPUT :
	*Y_x : Real component of Y.
	* Y_y : Imaginary component of Y.
	* X : Complex number X.
	*
	*OUTPUT :
	*hipFloatComplex : The result W = Y * X * .
	*/
	__device__ __forceinline__ hipFloatComplex complex_conj_mult_opt(float Y_x, float Y_y, hipFloatComplex X)
{
	hipFloatComplex W;

	// W_real = Y_real*X_real + Y_imag*X_imag
	// Zmiana __fmaf na **fmaf**
	W.x = fmaf(Y_y, X.y, Y_x * X.x); // FMA: (Y_y * X.y) + (Y_x * X.x)

	// W_imag = Y_imag*X_real - Y_real*X_imag
	// Zmiana __fmaf na **fmaf**
	W.y = fmaf(-Y_x, X.y, Y_y * X.x); // FMA: (-Y_x * X.y) + (Y_y * X.x)

	return W;
}


/**
 * KERNEL: ConvertAndScaleHIP
 *
 * DESCRIPTION: Converts raw interleaved float data (Real/Imag) into hipFloatComplex
 * format and applies a global scaling factor (SCALE).
 *
 * ... [Cia³o kernela bez zmian] ...
 */
  __global__ void ConvertAndScaleHIP(const float* __restrict__ raw_in, hipFloatComplex* __restrict__ complex_out, size_t Nth, float scale)
{
	// i - global thread index (index of the complex element to be processed)
	size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

	if (i >= Nth) return;

	// Conversion and Scaling
	// Zmiana __fmul_rn na mno¿enie (HIP mo¿e nie mieæ __fmul_rn)
	// Zmiana __ldg na prosty odczyt (hip nie ma __ldg w standardzie)
	complex_out[i].x = raw_in[2 * i] * scale;
	complex_out[i].y = raw_in[2 * i + 1] * scale;
}

/**
 * KERNEL: ShiftHIP
 *
 * ... [Cia³o kernela bez zmian] ...
 */
  __global__ void ShiftHIP(hipFloatComplex* __restrict__ In, hipFloatComplex* __restrict__ Out, size_t shift, size_t cudaN) {

	size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

	// Check if thread is within data bounds
	if (i >= cudaN) return;

	// Calculate the bitwise mask for modulo operation (assuming cudaN is power of 2)
	const size_t MASK = cudaN - 1;

	size_t j = i + shift;

	// Circular rotation (j = (i + shift) % cudaN;)
	j = j & MASK;

	// Read from global memory and write to Out[i]
	// Zmiana __ldg na prosty odczyt
	hipFloatComplex temp;
	temp.x = In[j].x;
	temp.y = In[j].y;

	Out[i] = temp;
}


/**
 * KERNEL: CorelateShiftHIP
 *
 * ... [Cia³o kernela bez zmian] ...
 */
  __global__ void CorelateShiftHIP(hipFloatComplex* __restrict__ BufX,
	hipFloatComplex* __restrict__ BufY,
	hipFloatComplex* __restrict__ BufW,
	size_t cudaCol, size_t cudaN)
{
	extern __shared__ hipFloatComplex shared_data[]; // Zmiana float2 na hipFloatComplex


#ifndef PRECOMPUTE
	// Load constant to shared memory once per block
	__shared__ float shared_zoom;
	if (hipThreadIdx_x == 0) {
		shared_zoom = constant_CudaZoom;
	}
#endif

	size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	size_t tid = hipThreadIdx_x;

	// 1. Load BufY to Shared Memory
	if (i < cudaN)
	{
		// Use vectorized load for better performance
		hipFloatComplex val = (BufY)[i];
		shared_data[tid].x = val.x;
		shared_data[tid].y = val.y;
	}

	__syncthreads();

	if (i >= cudaN) return;

	// 2. Pre-cache from Shared Memory to Registers
	const hipFloatComplex pre_casch_data = shared_data[tid];
	const float Y_x = pre_casch_data.x;
	const float Y_y = pre_casch_data.y;

	const size_t N = cudaN;
	const size_t MASK = N - 1;

	// 3. Loop Optimization
	// Aggressive unrolling
#pragma unroll(4)
	for (size_t n = 0; n < cudaCol; ++n)
	{
		// Calculate shifted index (circular shift)
#ifdef PRECOMPUTE
		size_t j_raw = i + constant_ShiftOffsets[n];
#else
		size_t j_raw = i + static_cast<size_t>(shared_zoom * n);
#endif

		//j = (j >= N) ? j - N : j;
		size_t j = j_raw & MASK;

		// Read BufX 
		// Zmiana __ldg na prosty odczyt
		hipFloatComplex val_x;
		val_x.x = BufX[j].x;
		val_x.y = BufX[j].y;

		// Compute correlation using inlined FMA function
		hipFloatComplex W_result = complex_conj_mult_opt(Y_x, Y_y, val_x);

		// Write to BufW
#ifdef PRECOMPUTE
		size_t output_index = i + constant_OutputOffsets[n];
#else
		size_t output_index = i + n * N;
#endif

		BufW[output_index] = W_result;
	}
}


/**
 * KERNEL: MagnitudeHIP_optimized
 *
 * DESCRIPTION: Computes the squared magnitude of the complex ambiguity map
 * (Inp) and stores the real result in the output buffer (Out).
 *
 * ... [Cia³o kernela] ...
 */
 __global__ void MagnitudeHIP_optimized(hipFloatComplex* Inp, float* Out, int cuda_row, int cuda_col, int cuda_shift, int cudaN) {
	// Calculate global thread indices
	size_t col_idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	size_t row_idx = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

	// Check if thread is within output matrix bounds
	if (col_idx < cuda_col && row_idx < cuda_row) {
		// Calculate input index in the large ambiguity map buffer
		size_t Inp_idx = row_idx + cuda_shift + col_idx * cudaN;

		// Boundary check protection
		if (Inp_idx >= (size_t)cudaN * cuda_col)
			// Inp_idx -= cudaN; // Index correction
			return;


		// Vectorized load 
		const hipFloatComplex complex_val = Inp[Inp_idx];

		// Calculate magnitude
#ifdef CALCULATE_SQRT_ON_CPU
		// U¿ycie fmaf zamiast __fmaf (zgodnoœæ z HIP)
		// Alternatywnie: complex_val.x * complex_val.x + complex_val.y * complex_val.y;
		Out[row_idx + col_idx * cuda_row] = fmaf(complex_val.x, complex_val.x, complex_val.y * complex_val.y);
#else
		float mag = fmaf(complex_val.x, complex_val.x, complex_val.y * complex_val.y);
		// Zmiana hip_rsqrtf na standardowe **sqrtf** (lub usuniêcie sqrtf, jeœli chodzi³o o kwadrat)
		// Pozostawiam pe³ne sqrtf dla zachowania logiki orygina³u (jeœli makro CALCULATE_SQRT_ON_CPU jest wy³¹czone)
		Out[row_idx + col_idx * cuda_row] = sqrtf(mag);
#endif

	}
}

