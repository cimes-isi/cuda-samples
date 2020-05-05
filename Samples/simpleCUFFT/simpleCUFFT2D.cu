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

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  printf("[simpleCUFFT2D] is starting...\n");

  int signal_size_x = 64;
  if (argc > 1) {
    signal_size_x = atoi(argv[1]);
  }
  int signal_size_y = signal_size_x;
  if (argc > 2) {
    signal_size_y = atoi(argv[2]);
  }
  printf("Signal size X = %zu\n", signal_size_x);
  printf("Signal size Y = %zu\n", signal_size_y);

  findCudaDevice(argc, (const char **)argv);

  // Allocate host memory for the signal
  const int mem_size = sizeof(Complex) * signal_size_x * signal_size_y;
  Complex *h_signal =
      reinterpret_cast<Complex *>(malloc(mem_size));

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < signal_size_x * signal_size_y; ++i) {
    h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
    h_signal[i].y = 0;
  }

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
      cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));

  // CUFFT plan simple API
  cufftHandle plan;
  checkCudaErrors(cufftPlan2d(&plan, signal_size_x, signal_size_y, CUFFT_C2C));

  // Record the start_kernel event
  checkCudaErrors(cudaEventRecord(start_kernel, NULL));

  // Transform signal
  printf("Transforming signal cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
                               reinterpret_cast<cufftComplex *>(d_signal),
                               CUFFT_FORWARD));

  // Check if kernel execution generated and error
  getLastCudaError("Kernel execution failed");

  // Transform signal back
  printf("Transforming signal back cufftExecC2C\n");
  checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
                               reinterpret_cast<cufftComplex *>(d_signal),
                               CUFFT_INVERSE));

  // Record the stop_kernel event
  checkCudaErrors(cudaEventRecord(stop_kernel, NULL));

  // Copy device memory to host
  checkCudaErrors(cudaMemcpy(h_signal, d_signal, mem_size,
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

  // Destroy CUFFT context
  checkCudaErrors(cufftDestroy(plan));

  // cleanup memory
  free(h_signal);
  checkCudaErrors(cudaFree(d_signal));

  exit(EXIT_SUCCESS);
}
