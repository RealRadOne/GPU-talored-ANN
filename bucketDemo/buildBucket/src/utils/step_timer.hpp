# pragma once
#include <chrono>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime.h>

template<typename StepBody>

double measure_step_seconds(StepBody&& step_body)
{
      const auto start_time = std::chrono::steady_clock::now();
      std::forward<StepBody>(step_body)();
      const cudaError_t sync_status = cudaDeviceSynchronize();
      if (sync_status != cudaSuccess) {
          throw std::runtime_error(std::string("measure_step_seconds: GPU work failed: ")
                                   + cudaGetErrorString(sync_status));
      }
      const auto end_time = std::chrono::steady_clock::now();
      return std::chrono::duration<double>(end_time - start_time).count();
}

inline void print_step_duration(const std::string& step_name, double elapsed_seconds)
{
      std::printf("  %s done [%.3fs]\n", step_name.c_str(), elapsed_seconds);
}

template<typename StepBody>
inline void run_step_and_print_duration(const std::string& step_name, StepBody&& step_body)
{
      print_step_duration(step_name, measure_step_seconds(std::forward<StepBody>(step_body)));
}

inline void log_step_timing(const std::string& output_dir, int iteration,const std::string& step_name, double seconds)
{
      print_step_duration(step_name, seconds);
      const std::string path = output_dir + "/step_timings.csv";
      const bool is_new_file = !std::ifstream(path).good();
      std::ofstream csv(path, std::ios::app);
      if (!csv) return;   // timing log is best-effort; never fail the pipeline
      if (is_new_file) csv << "unix_time_s,iteration,step,seconds\n";
      const double unix_time = std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      csv.setf(std::ios::fixed);
      csv.precision(3);
      csv << unix_time << ',' << iteration << ',' << step_name << ',' << seconds << '\n';
}