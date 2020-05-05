/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *, const Complex *,
                                                   int, float);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

#define DO_CHECK_RESULT 1

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  printf("[simpleCUFFT] is starting...\n");

  int signal_size = 64;
  if (argc > 1) {
    signal_size = atoi(argv[1]);
  }
// The filter size is assumed to be a number smaller than the signal size
  int filter_kernel_size = signal_size / 4;
  if (argc > 2) {
    filter_kernel_size = atoi(argv[2]);
  }
  printf("Signal size = %zu\n", signal_size);
  printf("Filter size = %zu\n", filter_kernel_size);

  findCudaDevice(argc, (const char **)argv);

  // Allocate host memory for the signal
  Complex *h_signal =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * signal_size));

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < signal_size; ++i) {
    h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
    h_signal[i].y = 0;
  }

  // Allocate host memory for the filter
  Complex *h_filter_kernel =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * filter_kernel_size));

  // Initialize the memory for the filter
  for (unsigned int i = 0; i < filter_kernel_size; ++i) {
    h_filter_kernel[i].x = rand() / static_cast<float>(RAND_MAX);
    h_filter_kernel[i].y = 0;
  }

  // Pad signal and filter kernel
  Complex *h_padded_signal;
  Complex *h_padded_filter_kernel;
  int new_size =
      PadData(h_signal, &h_padded_signal, signal_size, h_filter_kernel,
              &h_padded_filter_kernel, filter_kernel_size);
  int mem_size = sizeof(Complex) * new_size;

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start;
  checkCudaErrors(cudaEventCreate(&start));

  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEvent_t start_kernel;
  checkCudaErrors(cudaEventCreate(&start_kernel));

  cudaEvent_t stop_kernel;
  checkCudaErrors(cudaEventCreate(&stop_kernel));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, NULL));

  // Allocate device memory for signal
  Complex *d_signal;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size));
  // Copy host memory to device
  checkCudaErrors(
      cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));

  // Allocate device memory for filter kernel
  Complex *d_filter_kernel;
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&d_filter_kernel), mem_size));

  // Copy host memory to device
  checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
                             cudaMemcpyHostToDevice));

  // CUFFT plan simple API
  cufftHandle plan;
  checkCudaErrors(cufftPlan1d(&plan, new_size, CUFFT_C2C, 1));

  // CUFFT plan advanced API
  cufftHandle plan_adv;
  size_t workSize;
  long long int new_size_long = new_size;

  checkCudaErrors(cufftCreate(&plan_adv));
  checkCudaErrors(cufftXtMakePlanMany(plan_adv, 1, &new_size_long, NULL, 1, 1,
                                      CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, 1,
                                      &workSize, CUDA_C_32F));
  printf("Temporary buffer size %li bytes\n", workSize);

  // Record the start_kernel event
  checkCudaErrors(cudaEventRecord(start_kernel, NULL));

  // Transform signal and kernel
  printf("Transforming signal cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
                               reinterpret_cast<cufftComplex *>(d_signal),
                               CUFFT_FORWARD));
  checkCudaErrors(cufftExecC2C(
      plan_adv, reinterpret_cast<cufftComplex *>(d_filter_kernel),
      reinterpret_cast<cufftComplex *>(d_filter_kernel), CUFFT_FORWARD));

  // Multiply the coefficients together and normalize the result
  printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
  ComplexPointwiseMulAndScale<<<32, 256>>>(d_signal, d_filter_kernel, new_size,
                                           1.0f / new_size);

  // Check if kernel execution generated and error
  getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

  // Transform signal back
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
                               reinterpret_cast<cufftComplex *>(d_signal),
                               CUFFT_INVERSE));

  // Record the stop_kernel event
  checkCudaErrors(cudaEventRecord(stop_kernel, NULL));

  // Copy device memory to host
  Complex *h_convolved_signal = h_padded_signal;
  checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
                             cudaMemcpyDeviceToHost));

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, NULL));

  // Wait for the stop_kernel event to complete
  checkCudaErrors(cudaEventSynchronize(stop_kernel));
  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecKernel = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecKernel, start_kernel, stop_kernel));
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  printf("Kernel Time = %.3f msec\n", msecKernel);
  printf("Total Time  = %.3f msec\n", msecTotal);

  bool bTestResult = true;
#if DO_CHECK_RESULT
  // Allocate host memory for the convolution result
  Complex *h_convolved_signal_ref =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * signal_size));

  // Convolve on the host
  Convolve(h_signal, signal_size, h_filter_kernel, filter_kernel_size,
           h_convolved_signal_ref);

  // check result
  bTestResult = sdkCompareL2fe(
      reinterpret_cast<float *>(h_convolved_signal_ref),
      reinterpret_cast<float *>(h_convolved_signal), 2 * signal_size, 1e-5f);
  free(h_convolved_signal_ref);
#endif

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));
  checkCudaErrors(cufftDestroy(plan_adv));

  // cleanup memory
  free(h_signal);
  free(h_filter_kernel);
  free(h_padded_signal);
  free(h_padded_filter_kernel);
  checkCudaErrors(cudaFree(d_signal));
  checkCudaErrors(cudaFree(d_filter_kernel));

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

// Pad data
int PadData(const Complex *signal, Complex **padded_signal, int signal_size,
            const Complex *filter_kernel, Complex **padded_filter_kernel,
            int filter_kernel_size) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;
  int new_size = signal_size + maxRadius;

  // Pad signal
  Complex *new_data =
      reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
  memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
  memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
  *padded_signal = new_data;

  // Pad filter
  new_data = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
  memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
  memset(new_data + maxRadius, 0,
         (new_size - filter_kernel_size) * sizeof(Complex));
  memcpy(new_data + new_size - minRadius, filter_kernel,
         minRadius * sizeof(Complex));
  *padded_filter_kernel = new_data;

  return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// Computes convolution on the host
void Convolve(const Complex *signal, int signal_size,
              const Complex *filter_kernel, int filter_kernel_size,
              Complex *filtered_signal) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;

  // Loop over output element indices
  for (int i = 0; i < signal_size; ++i) {
    filtered_signal[i].x = filtered_signal[i].y = 0;

    // Loop over convolution indices
    for (int j = -maxRadius + 1; j <= minRadius; ++j) {
      int k = i + j;

      if (k >= 0 && k < signal_size) {
        filtered_signal[i] =
            ComplexAdd(filtered_signal[i],
                       ComplexMul(signal[k], filter_kernel[minRadius - j]));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                                   int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}
