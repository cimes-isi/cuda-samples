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

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

#define SIGNAL_SIZE_X 64
#define SIGNAL_SIZE_Y 64

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { runTest(argc, argv); }

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  printf("[simpleFFT2D] is starting...\n");

  // Allocate host memory for the signal
  const int mem_size = sizeof(Complex) * SIGNAL_SIZE_X * SIGNAL_SIZE_Y;
  Complex *h_signal = reinterpret_cast<Complex *>(fftwf_malloc(mem_size));

  // Initialize the memory for the signal
  for (unsigned int i = 0; i < SIGNAL_SIZE_X * SIGNAL_SIZE_Y; ++i) {
    // h_signal[i].x = rand() / (float)RAND_MAX;
    // h_signal[i].y = 0;
    h_signal[i] = rand() / (float)RAND_MAX + 0i;
  }

  Complex *h_signal_out = reinterpret_cast<Complex *>(fftwf_malloc(mem_size));
  Complex *h_signal_out_inv = reinterpret_cast<Complex *>(fftwf_malloc(mem_size));

  struct timespec ts_start;
  struct timespec ts_stop;
  struct timespec ts_kernel_start;
  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  // FFT plan simple API
  fftwf_plan plan = fftwf_plan_dft_2d(SIGNAL_SIZE_X, SIGNAL_SIZE_Y, h_signal, h_signal_out,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_plan plan_inv = fftwf_plan_dft_2d(SIGNAL_SIZE_X, SIGNAL_SIZE_Y, h_signal_out, h_signal_out_inv,
                                          FFTW_BACKWARD, FFTW_ESTIMATE);

  clock_gettime(CLOCK_MONOTONIC, &ts_kernel_start);

  // Transform signal and kernel
  printf("Transforming signal fftExecC2C\n");
  fftwf_execute(plan);

  // Transform signal back
  printf("Transforming signal back fftExecC2C\n");
  fftwf_execute(plan_inv);

  clock_gettime(CLOCK_MONOTONIC, &ts_stop);
  float ms = (float)((ts_stop.tv_sec - ts_start.tv_sec) * 1000) +
                     (ts_stop.tv_nsec - ts_start.tv_nsec) / 1000000.0f;
  float ms_kernel = (float)((ts_stop.tv_sec - ts_kernel_start.tv_sec) * 1000) +
                     (ts_stop.tv_nsec - ts_kernel_start.tv_nsec) / 1000000.0f;
  printf("Kernel Time = %.3f msec\n", ms_kernel);
  printf("Total Time = %.3f msec\n", ms);

  // Destroy FFT context
  fftwf_destroy_plan(plan);
  fftwf_destroy_plan(plan_inv);

  // cleanup memory
  fftwf_free(h_signal);
  fftwf_free(h_signal_out);
  fftwf_free(h_signal_out_inv);

  exit(EXIT_SUCCESS);
}
