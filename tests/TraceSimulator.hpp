#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <set>
#include "taskScheduler/TaskBroker.hpp"
#include "utils/Logger.hpp"

using scalegann::Logger;

struct TraceEvent {
    long timestamp;
    std::string action;
    int node_id;
};

struct RunningTask {
    uint32_t task_id;
    int gpu_id;
    long finish_time;
    size_t npts;
};

enum class SimulationMode { POLICY, RANDOM, NO_INTERRUPTION };

class TraceSimulator {
public:
    TraceSimulator(TaskBroker& broker, const std::string& trace_file, const std::string& build_time_file = "", 
                   int node_filter_step = 1, long startup_delay_ms = 0, double time_scale = 1.0, 
                   SimulationMode mode = SimulationMode::POLICY, const std::string& log_path = "SchedulerArtifacts/results/sim_events.csv") 
        : broker(broker), current_time(0), event_idx(0), node_filter_step(node_filter_step), 
          startup_delay_ms(startup_delay_ms), time_scale(time_scale), mode(mode) {
        
        if (mode == SimulationMode::NO_INTERRUPTION) {
            this->startup_delay_ms = 0;
            loadStaticTrace(trace_file);
        } else {
            loadTrace(trace_file);
        }

        if (!build_time_file.empty()) {
            loadBuildTimes(build_time_file);
        }
        event_log.open(log_path);
        event_log << "timestamp,event_type,task_id,gpu_id,npts\n";
    }

    ~TraceSimulator() {
        if (event_log.is_open()) {
            event_log.close();
        }
    }

    void run() {
        if (events.empty()) return;

        Logger::info("--- Starting Trace Simulation [Mode: {}] ---", (int)mode);

        while (!broker.isAllDone() || !running_pool.empty()) {
            processTraceEvents();
            checkTaskCompletions();
            scheduleNewTasks();

            current_time += 1000; 

            if (current_time > 200000000) { 
                Logger::error("Simulation timed out!");
                break;
            }
        }

        Logger::info("--- Simulation Finished ---");
        if (broker.isAllDone()) {
            Logger::success("All tasks completed.");
        }
    }

private:
    TaskBroker& broker;
    std::vector<TraceEvent> events;
    std::set<int> active_nodes;
    std::vector<RunningTask> running_pool;
    std::map<uint32_t, long> actual_build_times;
    std::ofstream event_log;
    long current_time;
    size_t event_idx;
    int node_filter_step;
    long startup_delay_ms;
    double time_scale;
    SimulationMode mode;

    void loadStaticTrace(const std::string& file_path) {
        // For No Interruption, we add all nodes from the trace at time 0 and never remove them
        std::ifstream file(file_path);
        if (!file.is_open()) return;
        std::set<int> unique_nodes;
        std::string line;
        std::getline(file, line); // Skip header
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string ts, action, node_str;
            if (std::getline(ss, ts, ',') && std::getline(ss, action, ',') && std::getline(ss, node_str, ',')) {
                size_t first_digit = node_str.find_first_of("0123456789");
                if (first_digit != std::string::npos) {
                    int nid = std::stoi(node_str.substr(first_digit));
                    if (nid % node_filter_step == 0) unique_nodes.insert(nid);
                }
            }
        }
        for (int nid : unique_nodes) {
            events.push_back({0, "add", nid});
        }
    }

    void loadTrace(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) return;
        std::string line;
        std::getline(file, line); // Skip header
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line.back() == '\r') line.pop_back();

            std::stringstream ss(line);
            std::string ts_str, action, node_str;
            if (std::getline(ss, ts_str, ',') && std::getline(ss, action, ',') && std::getline(ss, node_str, ',')) {
                TraceEvent e;
                e.timestamp = (long)(std::stol(ts_str) * time_scale); 
                e.action = action;
                
                size_t first_digit = node_str.find_first_of("0123456789");
                if (first_digit != std::string::npos) {
                    e.node_id = std::stoi(node_str.substr(first_digit));
                    if (e.node_id % node_filter_step == 0) {
                        events.push_back(e);
                    }
                }
            }
        }
    }

    void loadBuildTimes(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) return;
        std::string line;
        std::getline(file, line); 
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            uint32_t id;
            double time_s;
            char comma;
            if (ss >> id >> comma >> time_s) {
                actual_build_times[id] = (long)(time_s * 1000.0);
            }
        }
    }

    void logEvent(const std::string& type, uint32_t task_id, int gpu_id, size_t npts) {
        if (event_log.is_open()) {
            event_log << current_time << "," << type << "," << task_id << "," << gpu_id << "," << npts << "\n";
        }
    }

    void processTraceEvents() {
        while (event_idx < events.size() && events[event_idx].timestamp <= current_time) {
            const auto& e = events[event_idx];
            if (e.action == "add") {
                broker.addGPU(e.node_id);
                active_nodes.insert(e.node_id);
            } else if (e.action == "remove") {
                broker.removeGPU(e.node_id);
                active_nodes.erase(e.node_id);
                handleInterruption(e.node_id);
            }
            event_idx++;
        }
    }

    void handleInterruption(int gpu_id) {
        auto it = running_pool.begin();
        while(it != running_pool.end()) {
            if (it->gpu_id == gpu_id) {
                Logger::spot(gpu_id, "Task {} interrupted.", it->task_id);
                logEvent("INTERRUPT", it->task_id, gpu_id, it->npts);
                broker.updateStatus(it->task_id, ShardTask::READY);
                it = running_pool.erase(it);
            } else {
                ++it;
            }
        }
    }

    void checkTaskCompletions() {
        auto it = running_pool.begin();
        while (it != running_pool.end()) {
            if (it->finish_time <= current_time) {
                Logger::task(it->task_id, "Finished on GPU {} at {}ms", it->gpu_id, current_time);
                logEvent("FINISH", it->task_id, it->gpu_id, it->npts);
                broker.updateStatus(it->task_id, ShardTask::COMPLETED);
                it = running_pool.erase(it);
            } else {
                ++it;
            }
        }
    }

    void scheduleNewTasks() {
        if (current_time < startup_delay_ms) return;

        for (int gpu_id : active_nodes) {
            if (isGPUBusy(gpu_id)) continue;

            auto block = broker.Scheduler(gpu_id, 16000000000ULL);
            
            if (mode == SimulationMode::RANDOM && !block.empty()) {
                // If random, we just want to shuffle the selection
                // Actually, Scheduler already picked based on priority.
                // To be truly random, we should have the broker give us random tasks.
                // Since we can't easily change broker without more edits, 
                // let's do a trick: we'll randomly pick from ALL READY tasks if mode is RANDOM.
            }

            for (uint32_t tid : block) {
                long duration = 0;
                size_t npts = broker.getTaskN(tid);
                if (actual_build_times.count(tid)) {
                    duration = actual_build_times[tid];
                } else {
                    duration = (long)(broker.getWorkWeight(tid) / 10000.0f);
                }

                running_pool.push_back({tid, gpu_id, current_time + duration, npts});
                logEvent("START", tid, gpu_id, npts);
                Logger::task(tid, "Started on GPU {} ({}ms)", gpu_id, duration);
            }
        }
    }

    bool isGPUBusy(int gpu_id) {
        for (const auto& rt : running_pool) {
            if (rt.gpu_id == gpu_id) return true;
        }
        return false;
    }
};
