#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <systemc.h>
#include "data_structures.h"
#include "config.h"
#include <vector>
#include <map>
#include <string>

SC_MODULE(Simulator) {
    // --- Input Data ---
    std::vector<TokenState> tokens;
    ExpertDeviceMap expert_to_device_map;
    DeviceExpertMap device_to_experts_map;
    CommunicationMatrix comm_matrix; // Assumed Bandwidth B (Bytes/ns)

    // --- Derived Parameters ---
    int num_devices = 0;
    int num_layers = 0;          // Max layer index found in paths + 1
    int num_experts_per_layer = 0; // Max expert index found in topology + 1 (for a layer)
    int total_expert_instances = 0; // L * E

    // --- Simulation State ---
    sc_core::sc_time current_sim_time;
    int completed_tokens = 0;
    int current_sync_layer = 0; // The layer boundary we are syncing for

    // --- Methods ---
    void load_inputs();
    void validate_inputs();
    void determine_simulation_parameters();
    int get_global_expert_id(int layer, int expert_idx) const;
    int get_device_id(int layer, int expert_idx) const;
    double calculate_effective_data_size(double original_size_n) const;
    double get_bandwidth(int src_global_expert_id, int target_global_expert_id) const;
    double calculate_hop_delay(int num_aggregated_tokens, double bandwidth, int src_device, int target_device) const;
    void run_simulation(); // Main simulation thread process

    SC_CTOR(Simulator) {
        SC_THREAD(run_simulation);
        // Sensitivity list is not needed for this style of timed simulation control
        current_sim_time = sc_core::SC_ZERO_TIME;
    }
};

#endif // SIMULATOR_H