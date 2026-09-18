#pragma once

#include <cuda_runtime.h>

// Block size for scan kernels
constexpr int SCAN_BLOCK_SIZE = 512; // 512 threads can scan 1024 elements (2 per thread)

// Milestone 1: Hillis-Steele Scan (Step-efficient, Work-inefficient)
void launch_scan_hillis_steele(const float* d_in, float* d_out, int n);

// Milestone 2: Blelloch Scan (Work-efficient)
void launch_scan_blelloch(const float* d_in, float* d_out, int n);

// Milestone 3: Blelloch Scan with Bank Conflict Avoidance
void launch_scan_blelloch_padded(const float* d_in, float* d_out, int n);

// Status queries for the benchmark harness
bool is_hillis_steele_implemented();
bool is_blelloch_implemented();
bool is_blelloch_padded_implemented();
