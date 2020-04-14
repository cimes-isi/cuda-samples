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

/* Example showing the use of FFT for fast 1D-convolution using FFTW. */

#include <complex.h>

// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// includes, project
#include <fftw3.h>

// Complex data type
typedef float complex Complex;
static inline Complex ComplexScale(Complex, float);
static inline Complex ComplexMul(Complex, Complex);
static void ComplexPointwiseMulAndScale(Complex *, const Complex *, int, float);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE 50
#define FILTER_KERNEL_SIZE 11

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  printf("[simpleFFT] is starting...\n");

  // Allocate host memory for the signal
  Complex *h_signal = reinterpret_cast<Complex *>(fftwf_malloc(sizeof(Complex) * SIGNAL_SIZE));

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
    // h_signal[i].x = rand() / (float)RAND_MAX;
    // h_signal[i].y = 0;
    h_signal[i] = rand() / (float)RAND_MAX + 0i;
  }

  // Allocate host memory for the filter
  Complex *h_filter_kernel = reinterpret_cast<Complex *>(fftwf_malloc(sizeof(Complex) * FILTER_KERNEL_SIZE));

  // Initialize the memory for the filter
  for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
    h_signal[i] = rand() / (float)RAND_MAX + 0i;
    // h_filter_kernel[i].x = rand() / (float)RAND_MAX;
    // h_filter_kernel[i].y = 0;
  }

  // Pad signal and filter kernel
  Complex *h_padded_signal;
  Complex *h_padded_filter_kernel;
  int new_size =
      PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
              &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
  int mem_size = sizeof(Complex) * new_size;
  Complex *h_padded_signal_out = reinterpret_cast<Complex *>(fftwf_malloc(mem_size));
  Complex *h_padded_signal_out_inv = reinterpret_cast<Complex *>(fftwf_malloc(mem_size));
  Complex *h_padded_filter_kernel_out = reinterpret_cast<Complex *>(fftwf_malloc(mem_size));

  struct timespec ts_start;
  struct timespec ts_stop;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  // FFT plan simple API
  fftwf_plan plan = fftwf_plan_dft_1d(new_size, h_padded_signal, h_padded_signal_out,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_plan plan_inv = fftwf_plan_dft_1d(new_size, h_padded_signal_out, h_padded_signal_out_inv,
                                          FFTW_BACKWARD, FFTW_ESTIMATE);

  // FFT plan advanced API
  fftwf_plan plan_adv = fftwf_plan_many_dft(1, &new_size, 1,
                                            h_padded_filter_kernel, NULL, 1, 1,
                                            h_padded_filter_kernel_out, NULL, 1, 1,
                                            FFTW_FORWARD, FFTW_ESTIMATE);

  // Transform signal and kernel
  printf("Transforming signal fftExecC2C\n");
  fftwf_execute(plan);
  fftwf_execute(plan_adv);

  // Multiply the coefficients together and normalize the result
  printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
  ComplexPointwiseMulAndScale(h_padded_signal_out, h_padded_filter_kernel_out, new_size,
                              1.0f / new_size);

  // Transform signal back
  printf("Transforming signal back fftExecC2C\n");
  fftwf_execute(plan_inv);

  clock_gettime(CLOCK_MONOTONIC, &ts_stop);
  float ms = (float)((ts_stop.tv_sec - ts_start.tv_sec) * 1000) +
                     (ts_stop.tv_nsec - ts_start.tv_nsec) / 1000000.0f;
  printf("Time = %.3f msec\n", ms);

  // Destroy FFT context
  fftwf_destroy_plan(plan);
  fftwf_destroy_plan(plan_inv);
  fftwf_destroy_plan(plan_adv);

  // cleanup memory
  fftwf_free(h_signal);
  fftwf_free(h_filter_kernel);
  fftwf_free(h_padded_signal);
  fftwf_free(h_padded_filter_kernel);
  fftwf_free(h_padded_signal_out);
  fftwf_free(h_padded_signal_out_inv);
  fftwf_free(h_padded_filter_kernel_out);

  exit(EXIT_SUCCESS);
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
      reinterpret_cast<Complex *>(fftwf_malloc(sizeof(Complex) * new_size));
  memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
  memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
  *padded_signal = new_data;

  // Pad filter
  new_data = reinterpret_cast<Complex *>(fftwf_malloc(sizeof(Complex) * new_size));
  memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
  memset(new_data + maxRadius, 0,
         (new_size - filter_kernel_size) * sizeof(Complex));
  memcpy(new_data + new_size - minRadius, filter_kernel,
         minRadius * sizeof(Complex));
  *padded_filter_kernel = new_data;

  return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex scale
static inline Complex ComplexScale(Complex a, float s) {
  // Complex c;
  // c.x = s * a.x;
  // c.y = s * a.y;
  // return c;
  return s * a;
}

// Complex multiplication
static inline Complex ComplexMul(Complex a, Complex b) {
  // Complex c;
  // c.x = a.x * b.x - a.y * b.y;
  // c.y = a.x * b.y + a.y * b.x;
  // return c;
  return a * b;
}

// Complex pointwise multiplication
static void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                        int size, float scale) {
  // const int numThreads = blockDim.x * gridDim.x;
  // const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  // const int numThreads = creal(blockDim) * creal(gridDim);
  // const int threadID = creal(blockIdx) * creal(blockDim) + creal(threadIdx);
  const int numThreads = 1;
  const int threadID = 0;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}
