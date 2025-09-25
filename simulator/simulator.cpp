#include "simulator.h"
#include "parser.h"
#include "config.h"
#include <iostream>
#include <stdexcept>
#include <cmath>     // ceil, max
#include <limits>    // numeric_limits
#include <map>       // for aggregation map

// Helper to get global expert ID
int Simulator::get_global_expert_id(int layer, int expert_idx) const {
    if (layer < 0 || expert_idx < 0 || num_experts_per_layer <= 0) {
        // Handle the initial -1:-1 state or invalid cases
        return -1;
    }
    if (layer >= num_layers || expert_idx >= num_experts_per_layer) {
        SC_REPORT_WARNING("ID_MAP", ("Expert ID request out of bounds: Layer " + std::to_string(layer)
                         + ", Expert " + std::to_string(expert_idx)).c_str());
        // Decide how to handle out-of-bounds - return -1 or throw?
        return -1; // Indicate invalid ID
    }
    return layer * num_experts_per_layer + expert_idx;
}

// Helper to get device ID for a given expert instance
int Simulator::get_device_id(int layer, int expert_idx) const {
     // Special case for the initial state before layer 0
    if (layer == -1 && expert_idx == -1) {
        return -1; // Represents "source" before entering the system
    }

    int global_expert_id = -1;
    // Find which global expert ID corresponds to this expert_idx *within its layer*.
    // This requires knowing the topology mapping. We assume expert_idx is the *local* index on the device/layer.
    // Re-interpreting: The routing `L:E` likely means Layer L, *Global Expert Index* E (0 to E_total-1).
    // Let's refine `get_global_expert_id` - it's just the expert index E if E is globally unique per layer.
    // Re-re-interpreting: `topology.txt` maps *absolute* expert IDs (0 to 63 in example) to devices.
    // `routing.txt` L:E means Layer L, Expert *instance* E at that layer.
    // *** Clarification needed: Is 'E' in routing.txt a global ID (0-63) or a per-layer index (0-7)? ***
    // --> Assuming 'E' in routing.txt is the *global* expert ID (0-63 type).
    // --> Assuming `communication.txt` is indexed by `[src_layer * E_per_layer + src_E_idx]`... NO, this is inconsistent.
    // --> Let's assume `communication.txt` size is `(L*E_total) x (L*E_total)` is WRONG.
    // --> Let's assume `communication.txt` size is `E_total x E_total` and represents communication *between the physical experts*, ignoring layers. This doesn't capture layer hops well.
    // --> MOST LIKELY: `communication.txt` is `(L*E) x (L*E)` where E is Experts *per layer*, and the index is `layer*E + expert_index_within_layer`.
    // --> AND `routing.txt` L:E uses `expert_index_within_layer`.
    // --> AND `topology.txt` maps `expert_index_within_layer` to `device_id`. THIS IS AMBIGUOUS. How do we know which layer an expert in topology.txt belongs to?

    // *** Final Chosen Interpretation (Needs Verification) ***
    // 1. `topology.txt`: Defines GLOBAL expert IDs (0..N_experts_total-1) and maps them to devices.
    // 2. `routing.txt`: L:E refers to Layer L, GLOBAL expert ID E.
    // 3. `communication.txt`: Matrix size `M x M`. `M` must be `num_layers * num_experts_per_layer`? No, M must be `num_layers * N_experts_total`? This seems too large.
    // Let's revert to the interpretation where `communication.txt` is `(L*E) x (L*E)` where E is experts *per layer*.
    // And `routing.txt` L:E uses E as expert index *within layer* (0 to E-1).
    // And `topology.txt` must somehow imply the per-layer mapping or be structured differently. The example `device0: 0..7` suggests experts 0..7 are the experts *for each layer* on device 0.

    // *** Revised Implementation based on L*E matrix and per-layer expert index ***
    global_expert_id = get_global_expert_id(layer, expert_idx); // Calculate flat index L*E + e

    // Now, how to map this flat index (layer, expert_in_layer) to a device?
    // The topology file needs re-interpretation. Assume `topology.txt` `deviceX: e1, e2...` means
    // that device X hosts experts with indices e1, e2... *at every layer*.
    // Find which device hosts `expert_idx`.
    for (const auto& pair : device_to_experts_map) {
        int device_id = pair.first;
        const auto& experts_on_device = pair.second;
        for (int hosted_expert_idx : experts_on_device) {
            if (hosted_expert_idx == expert_idx) {
                return device_id;
            }
        }
    }

    // If not found (shouldn't happen with valid inputs)
     SC_REPORT_ERROR("DEVICE_MAP", ("Expert index " + std::to_string(expert_idx) + " not found in topology map.").c_str());
    return -1; // Error
}


void Simulator::load_inputs() {
    try {
        tokens = Parser::parse_routing(config::ROUTING_FILE);
        Parser::parse_topology(config::TOPOLOGY_FILE, expert_to_device_map, device_to_experts_map, num_devices);
        // Pass pointers to allow parser to update L and E based on other files if needed
        comm_matrix = Parser::parse_communication(config::COMMUNICATION_FILE, num_layers, num_experts_per_layer);

         if (tokens.empty()) {
            throw std::runtime_error("No valid token paths found in routing file.");
         }
          if (num_devices == 0) {
            throw std::runtime_error("No devices found in topology file.");
         }
         if (comm_matrix.empty()) {
             throw std::runtime_error("Communication matrix is empty.");
         }

    } catch (const std::exception& e) {
        SC_REPORT_FATAL("INPUT_ERROR", ("Failed to load or parse input files: " + std::string(e.what())).c_str());
    }
    std::cout << "Info: Successfully parsed input files." << std::endl;
    std::cout << "Info: Found " << tokens.size() << " tokens, " << num_devices << " devices." << std::endl;
}

void Simulator::determine_simulation_parameters() {
    // Determine num_layers and num_experts_per_layer more robustly
    int max_layer = -1;
    int max_expert_idx = -1; // Expert index *within* a layer

    for(const auto& token : tokens) {
        for(const auto& segment : token.path) {
            if(segment.source.layer > max_layer) max_layer = segment.source.layer;
            if(segment.target.layer > max_layer) max_layer = segment.target.layer;
            // Assuming E in L:E is expert index within layer
             if(segment.source.expert > max_expert_idx) max_expert_idx = segment.source.expert;
             if(segment.target.expert > max_expert_idx) max_expert_idx = segment.target.expert;
        }
    }
    num_layers = max_layer + 1; // e.g., max layer 4 -> 5 layers (0, 1, 2, 3, 4)

    // Find max expert index from topology (assuming indices are per-layer)
    int max_topo_expert_idx = -1;
    for(const auto& pair : device_to_experts_map) {
        for(int expert_idx : pair.second) {
            if (expert_idx > max_topo_expert_idx) max_topo_expert_idx = expert_idx;
        }
    }
     num_experts_per_layer = max_topo_expert_idx + 1;

    std::cout << "Info: Determined Sim Params: num_layers = " << num_layers
              << ", num_experts_per_layer = " << num_experts_per_layer << std::endl;

     total_expert_instances = num_layers * num_experts_per_layer;

    // Validate communication matrix size
    if (comm_matrix.size() != total_expert_instances) {
         std::string error_msg = "Communication matrix size mismatch. Expected " +
                                std::to_string(total_expert_instances) + "x" + std::to_string(total_expert_instances) +
                                " (L=" + std::to_string(num_layers) + ", E=" + std::to_string(num_experts_per_layer) +
                                "), but got " + std::to_string(comm_matrix.size()) + "x" + (comm_matrix.empty() ? "0" : std::to_string(comm_matrix[0].size()));
        SC_REPORT_FATAL("MATRIX_SIZE", error_msg.c_str());
    }
}


void Simulator::validate_inputs() {
     // Check if all experts mentioned in routing exist in topology and comm matrix indices
     for(const auto& token : tokens) {
         for(const auto& segment : token.path) {
             // Check source (ignore -1:-1)
             if (segment.source.layer >= 0) {
                 if (segment.source.layer >= num_layers)
                    SC_REPORT_FATAL("VALIDATION", ("Token " + std::to_string(token.id) + " uses layer " + std::to_string(segment.source.layer) + " which exceeds max layer " + std::to_string(num_layers-1)).c_str());
                 if (segment.source.expert < 0 || segment.source.expert >= num_experts_per_layer)
                    SC_REPORT_FATAL("VALIDATION", ("Token " + std::to_string(token.id) + " uses expert " + std::to_string(segment.source.expert) + " which is out of range [0," + std::to_string(num_experts_per_layer-1) + "]").c_str());
                 if (get_device_id(segment.source.layer, segment.source.expert) < 0)
                     SC_REPORT_FATAL("VALIDATION", ("Token " + std::to_string(token.id) + ": Expert " + std::to_string(segment.source.expert) + " not found in topology.").c_str());
             }
             // Check target
              if (segment.target.layer >= num_layers)
                  SC_REPORT_FATAL("VALIDATION", ("Token " + std::to_string(token.id) + " targets layer " + std::to_string(segment.target.layer) + " which exceeds max layer " + std::to_string(num_layers-1)).c_str());
              if (segment.target.expert < 0 || segment.target.expert >= num_experts_per_layer)
                  SC_REPORT_FATAL("VALIDATION", ("Token " + std::to_string(token.id) + " targets expert " + std::to_string(segment.target.expert) + " which is out of range [0," + std::to_string(num_experts_per_layer-1) + "]").c_str());
              if (get_device_id(segment.target.layer, segment.target.expert) < 0)
                  SC_REPORT_FATAL("VALIDATION", ("Token " + std::to_string(token.id) + ": Target Expert " + std::to_string(segment.target.expert) + " not found in topology.").c_str());
         }
     }
     std::cout << "Info: Input validation successful." << std::endl;
}


double Simulator::calculate_effective_data_size(double original_size_n) const {
    if (config::MAX_PAYLOAD <= 0) {
        SC_REPORT_WARNING("CALC", "MaxPayload is zero or negative, returning original size.");
        return original_size_n;
    }
    double num_packets = std::ceil(original_size_n / config::MAX_PAYLOAD);
    return num_packets * config::FLIT_SIZE + original_size_n;
}

double Simulator::get_bandwidth(int src_global_expert_id, int target_global_expert_id) const {
     if (src_global_expert_id < 0 || src_global_expert_id >= total_expert_instances ||
         target_global_expert_id < 0 || target_global_expert_id >= total_expert_instances)
     {
          SC_REPORT_WARNING("BW_LOOKUP", ("Bandwidth lookup index out of bounds: src="
                            + std::to_string(src_global_expert_id) + ", tgt="
                            + std::to_string(target_global_expert_id)).c_str());
         return 0.0; // Indicate error or no path
     }
     return comm_matrix[src_global_expert_id][target_global_expert_id];
}


double Simulator::calculate_hop_delay(int num_aggregated_tokens, double bandwidth, int src_device, int target_device) const {
    if (num_aggregated_tokens <= 0) return 0.0;

    // Handle intra-device communication
    if (src_device == target_device) {
        return config::ENABLE_INTRA_DEVICE_LATENCY ? config::INTRA_DEVICE_LATENCY : 0.0;
    }

    // Handle inter-device communication
    if (bandwidth <= 1e-9) { // Check for zero or near-zero bandwidth
        SC_REPORT_WARNING("DELAY_CALC", ("Zero or very low bandwidth (" + std::to_string(bandwidth)
                          + ") between devices " + std::to_string(src_device)
                          + " and " + std::to_string(target_device) + ". Assuming infinite delay.").c_str());
        return std::numeric_limits<double>::infinity();
    }

    double total_original_data = num_aggregated_tokens * config::DATA_SIZE_N;
    double effective_data = calculate_effective_data_size(total_original_data);

    double transmission_delay = effective_data / bandwidth;
    double total_delay = config::LINK_LATENCY_L + config::OVERHEAD_O + transmission_delay;

    return total_delay; // In simulation time units (e.g., ns)
}


void Simulator::run_simulation() {
    std::cout << "\n--- Starting MoE Inference Simulation ---" << std::endl;
    load_inputs();
    determine_simulation_parameters();
    validate_inputs();

    current_sim_time = sc_core::SC_ZERO_TIME;
    completed_tokens = 0;
    int simulation_step = 0;

    while (completed_tokens < tokens.size()) {
        std::cout << "\n--- Simulation Step " << simulation_step
                  << " (Sim Time: " << sc_core::sc_time_stamp() << ") ---" << std::endl;

        // Map: Key=(src_device, target_device), Value=vector of token IDs making this hop
        std::map<std::pair<int, int>, std::vector<int>> active_hops;
        // Map: Key=token_id, Value=max delay encountered by this token in this step
        std::map<int, double> token_delays_this_step;
        // Map: Key=token_id, Value=the specific hop segment causing the delay
         std::map<int, PathSegment> token_segment_this_step;

        bool tokens_advanced = false;

        // 1. Identify next hops for all active tokens
        for (auto& token : tokens) {
            if (token.finished) continue;

            const PathSegment& segment = token.path[token.current_segment_index];
            token_segment_this_step[token.id] = segment; // Store the hop for this step

            int src_layer = segment.source.layer;
            int src_expert = segment.source.expert;
            int target_layer = segment.target.layer;
            int target_expert = segment.target.expert;

            int src_device = get_device_id(src_layer, src_expert);
            int target_device = get_device_id(target_layer, target_expert);

             // Debug print
             // std::cout << "  Token " << token.id << ": Prep hop " << src_layer << ":" << src_expert
             //           << " (Dev " << src_device << ") -> " << target_layer << ":" << target_expert
             //           << " (Dev " << target_device << ")" << std::endl;


            if (src_device < 0 && src_layer != -1) { // Error getting source device
                 SC_REPORT_ERROR("SIM_STEP", ("Could not find source device for token " + std::to_string(token.id)
                                 + " at " + std::to_string(src_layer) + ":" + std::to_string(src_expert)).c_str());
                 continue;
            }
             if (target_device < 0) { // Error getting target device
                  SC_REPORT_ERROR("SIM_STEP", ("Could not find target device for token " + std::to_string(token.id)
                                  + " target " + std::to_string(target_layer) + ":" + std::to_string(target_expert)).c_str());
                 continue;
            }

            active_hops[{src_device, target_device}].push_back(token.id);
        }

        if (active_hops.empty() && completed_tokens < tokens.size()) {
             SC_REPORT_WARNING("SIM_STEP", "No active hops found, but not all tokens finished. Check logic.");
             break; // Avoid infinite loop
        }


        // 2. Calculate delays for aggregated hops
        double max_delay_this_step = 0.0;
        std::map<std::pair<int, int>, double> hop_delays; // Store calculated delay for each (src,tgt) device pair

        for (const auto& pair : active_hops) {
            int src_device = pair.first.first;
            int target_device = pair.first.second;
            const auto& token_ids_on_hop = pair.second;
            int num_tokens = token_ids_on_hop.size();

            if (num_tokens == 0) continue;

            double hop_bandwidth = 0.0;
            if (src_device != target_device) {
                // Determine bandwidth B for this device pair (src_dev -> target_dev)
                // This requires a strategy: min/max/avg bandwidth between experts?
                // Simplification: Use the bandwidth of the *first* token's specific expert hop
                // A more complex model might consider link capacity.
                const auto& first_token_segment = token_segment_this_step[token_ids_on_hop[0]];
                int src_global_id = get_global_expert_id(first_token_segment.source.layer, first_token_segment.source.expert);
                int target_global_id = get_global_expert_id(first_token_segment.target.layer, first_token_segment.target.expert);
                hop_bandwidth = get_bandwidth(src_global_id, target_global_id);

                if (hop_bandwidth <= 1e-9) {
                     SC_REPORT_WARNING("Bandwidth", ("Zero or low bandwidth for hop: " +
                         std::to_string(first_token_segment.source.layer) + ":" + std::to_string(first_token_segment.source.expert) + " -> " +
                         std::to_string(first_token_segment.target.layer) + ":" + std::to_string(first_token_segment.target.expert)).c_str());
                 }

            } else {
                 // Intra-device doesn't strictly need bandwidth for latency calculation in this model
                 hop_bandwidth = std::numeric_limits<double>::infinity(); // Indicate not used for delay formula
            }


            int effective_num_tokens = config::ENABLE_DATA_AGGREGATION ? num_tokens : 1;
             // Note: If aggregation is off, we should calculate delay for each token individually
             // and the step delay is the max over all individual token delays.
             // The current calculation assumes aggregation applies to the *link*, affecting all tokens using it.

            double delay = calculate_hop_delay(num_tokens, hop_bandwidth, src_device, target_device);
             hop_delays[pair.first] = delay; // Store delay for this device-pair link


             std::cout << "  Hop: Dev " << src_device << " -> Dev " << target_device
                       << ", Tokens: " << num_tokens << ", BW: " << hop_bandwidth
                       << ", Delay: " << delay << " ns" << std::endl;


            // Update max delay for the step
             max_delay_this_step = std::max(max_delay_this_step, delay);

             // Track the delay associated with each token using this hop
             for(int token_id : token_ids_on_hop) {
                 token_delays_this_step[token_id] = std::max(token_delays_this_step[token_id], delay);
             }

        }


        if (max_delay_this_step == 0 && completed_tokens < tokens.size()) {
            // If max delay is zero, advance time minimally if tokens still need to move
            // This can happen with only intra-device hops if latency is disabled.
             bool any_active = false;
             for(const auto& token : tokens) if (!token.finished) any_active = true;
             if (any_active) {
                 // std::cout << "  Zero delay step, advancing time minimally." << std::endl;
                 // wait(sc_core::sc_get_time_resolution()); // Advance by smallest possible time
                 // OR just proceed to update state without waiting
             } else {
                 // All tokens finished, delay calculation might not have run
             }
        } else if (max_delay_this_step > 0) {
             // 3. Advance SystemC time
             sc_core::sc_time delay_amount(max_delay_this_step, config::TIME_UNIT);
             wait(delay_amount);
             current_sim_time = sc_core::sc_time_stamp();
             std::cout << "  Advancing time by " << delay_amount << ". New Sim Time: " << current_sim_time << std::endl;
        }


        // 4. Update token states for tokens whose hops completed
        for (auto& token : tokens) {
            if (token.finished) continue;

             // Check if this token was part of the completed step
             // A token completes its hop if the simulation time advances
             // (or if delay was 0 and it had an active hop)
             bool hop_considered = token_segment_this_step.count(token.id);

             if(hop_considered) { // Only advance tokens that were processed this step
                 token.current_location = token.path[token.current_segment_index].target;
                 token.current_segment_index++;
                 tokens_advanced = true;

                 std::cout << "    Token " << token.id << " advanced to Layer " << token.current_location.layer
                           << ", Expert " << token.current_location.expert << std::endl;


                 if (token.current_segment_index >= token.path.size()) {
                     token.finished = true;
                     completed_tokens++;
                     std::cout << "    Token " << token.id << " Finished! (Total Time: " << current_sim_time << ")" << std::endl;
                 }
             }
        }

         if (!tokens_advanced && completed_tokens < tokens.size()) {
             SC_REPORT_WARNING("SIM_LOOP", "No tokens advanced in a step, but simulation not finished. Check for deadlocks or zero-delay loops.");
             // You might want to break here to prevent infinite loops in certain scenarios
              break;
         }


        simulation_step++;
        if(simulation_step > 10000) { // Safety break
             SC_REPORT_ERROR("SIM_LOOP", "Simulation exceeded maximum steps. Possible infinite loop.");
             break;
        }

    } // End while loop

    std::cout << "\n--- Simulation Finished ---" << std::endl;
    std::cout << "Total Simulation Time: " << sc_core::sc_time_stamp() << std::endl;
    std::cout << "Completed Tokens: " << completed_tokens << "/" << tokens.size() << std::endl;
}