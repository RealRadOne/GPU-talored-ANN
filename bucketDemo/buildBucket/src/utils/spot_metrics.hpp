#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

enum class PipelineStep : int {
    kLoadSample         = 1,
    kSelectCentroids    = 2,
    kBuildCentroidGraph = 3,
    kAssignVectors      = 4,
    kWriteBuckets       = 5,
    kBuildVectorKnn     = 6,
    kReorderBuckets     = 7,
};

struct ProgressRecord {
    int iteration           = 0;
    PipelineStep step       = PipelineStep::kLoadSample;
    int64_t completed_units = 0;
    int64_t total_units     = 1;
    double elapsed_seconds  = 0.0;
};

struct PreviousRunSummary {
    bool was_interrupted = false;
    ProgressRecord last_record;
};

inline constexpr const char* kRunStartLine    = "=== Pipeline run started ===";
inline constexpr const char* kRunCompleteLine = "=== Pipeline run completed ===";

inline bool has_reached(const ProgressRecord& current, const ProgressRecord& target) {
    if (current.iteration != target.iteration) return current.iteration > target.iteration;
    if (current.step != target.step) return current.step > target.step;
    return current.completed_units * target.total_units
        >= target.completed_units * current.total_units;
}

inline std::optional<ProgressRecord> parse_progress_line(const std::string& line) {
    ProgressRecord record;
    int step_number = 0;
    char separator = ',';
    std::istringstream fields(line);
    fields >> record.elapsed_seconds >> separator >> record.iteration >> separator
           >> step_number >> separator >> record.completed_units >> separator
           >> record.total_units;
    if (!fields) return std::nullopt;
    record.step = static_cast<PipelineStep>(step_number);
    return record;
}

inline PreviousRunSummary read_previous_run_summary(const std::string& progress_path) {
    PreviousRunSummary summary;
    bool has_record = false;
    bool is_complete = false;
    std::ifstream progress_file(progress_path);
    std::string line;
    while (std::getline(progress_file, line)) {
        if (line == kRunStartLine) {
            has_record = false;
            is_complete = false;
        } else if (line == kRunCompleteLine) {
            is_complete = true;
        } else if (auto record = parse_progress_line(line)) {
            summary.last_record = *record;
            has_record = true;
        }
    }
    summary.was_interrupted = has_record && !is_complete;
    return summary;
}

class RestartProgressLog {
public:
    explicit RestartProgressLog(const std::string& output_dir)
        : progress_path_(output_dir + "/restart_progress.csv"),
          previous_run_(read_previous_run_summary(progress_path_)),
          start_time_(std::chrono::steady_clock::now())
    {
        progress_file_.open(progress_path_, std::ios::app);
        if (!progress_file_) {
            throw std::runtime_error("RestartProgressLog: cannot open " + progress_path_);
        }
        write_line(kRunStartLine);
    }

    void set_iteration(int iteration) { iteration_ = iteration; }

    void record_step_complete(PipelineStep step) { record_progress(step, 1, 1); }

    void record_progress(PipelineStep step, int64_t completed_units, int64_t total_units) {
        const ProgressRecord record{iteration_, step, completed_units,
                                    std::max<int64_t>(1, total_units), elapsed_seconds()};
        std::ostringstream line;
        line << record.elapsed_seconds << ',' << record.iteration << ','
             << static_cast<int>(record.step) << ',' << record.completed_units << ','
             << record.total_units;
        write_line(line.str());
        update_recovery_latency(record);
    }

    void record_run_complete() { write_line(kRunCompleteLine); }

    const PreviousRunSummary& previous_run() const { return previous_run_; }

    std::optional<double> recovery_latency_seconds() const { return recovery_latency_seconds_; }

private:
    double elapsed_seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

    void write_line(const std::string& line) {
        progress_file_ << line << '\n' << std::flush;
        if (!progress_file_) {
            throw std::runtime_error("RestartProgressLog: write failed: " + progress_path_);
        }
    }

    void update_recovery_latency(const ProgressRecord& record) {
        const bool is_first_regain = previous_run_.was_interrupted && !recovery_latency_seconds_
                                     && has_reached(record, previous_run_.last_record);
        if (is_first_regain) recovery_latency_seconds_ = record.elapsed_seconds;
    }

    std::string                           progress_path_;
    PreviousRunSummary                    previous_run_;
    std::chrono::steady_clock::time_point start_time_;
    std::ofstream                         progress_file_;
    int                                   iteration_ = 0;
    std::optional<double>                 recovery_latency_seconds_;
};

inline void print_interrupted_run(const PreviousRunSummary& previous_run) {
    if (!previous_run.was_interrupted) return;
    const ProgressRecord& last_record = previous_run.last_record;
    std::printf("  [restart] previous run was interrupted at iteration %d, step %d "
                "(%lld/%lld) after %.1fs\n",
                last_record.iteration, static_cast<int>(last_record.step),
                static_cast<long long>(last_record.completed_units),
                static_cast<long long>(last_record.total_units), last_record.elapsed_seconds);
}

inline void print_rows_lost_on_restart(int64_t rows_before_restart, int64_t rows_after_restart,
                                       int64_t total_rows) {
    if (rows_before_restart == 0) return;
    const int64_t rows_lost = std::max<int64_t>(0, rows_before_restart - rows_after_restart);
    std::printf("  [restart] result rows on disk: %lld / %lld, kept: %lld, lost: %lld (%.1f%%)\n",
                static_cast<long long>(rows_before_restart), static_cast<long long>(total_rows),
                static_cast<long long>(rows_after_restart), static_cast<long long>(rows_lost),
                100.0 * static_cast<double>(rows_lost) / static_cast<double>(rows_before_restart));
}

inline void print_recovery_latency(const RestartProgressLog& progress_log) {
    const std::optional<double> latency_seconds = progress_log.recovery_latency_seconds();
    if (!latency_seconds) return;
    std::printf("  [restart] recovery latency: %.3fs (the interrupted run had run %.3fs)\n",
                *latency_seconds, progress_log.previous_run().last_record.elapsed_seconds);
}

inline int64_t count_rows_with_neighbors(const std::string& knn_path) {
    std::ifstream knn_file(knn_path, std::ios::binary);
    if (!knn_file) return 0;
    int64_t row_count = 0;
    int32_t neighbors_per_row = 0;
    knn_file.read(reinterpret_cast<char*>(&row_count), sizeof(row_count));
    knn_file.read(reinterpret_cast<char*>(&neighbors_per_row), sizeof(neighbors_per_row));
    if (!knn_file || row_count <= 0 || neighbors_per_row <= 0) {
        throw std::runtime_error("RunningKnnFile: invalid header in " + knn_path);
    }
    const auto row_bytes =
        static_cast<std::streamsize>(static_cast<size_t>(neighbors_per_row) * sizeof(int32_t));
    std::vector<int32_t> row(static_cast<size_t>(neighbors_per_row));
    int64_t rows_with_neighbors = 0;
    for (int64_t row_index = 0; row_index < row_count; ++row_index) {
        if (!knn_file.read(reinterpret_cast<char*>(row.data()), row_bytes)) break;
        if (row[0] >= 0) ++rows_with_neighbors;
    }
    return rows_with_neighbors;
}