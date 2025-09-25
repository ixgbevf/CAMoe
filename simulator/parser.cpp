#include "parser.h"
#include "data_structures.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath> // For sqrt

namespace Parser {

    Hop parse_hop(const std::string& hop_str) {
        size_t colon_pos = hop_str.find(':');
        if (colon_pos == std::string::npos) {
            throw ParseError("Invalid hop format: '" + hop_str + "'. Expected 'layer:expert'.");
        }
        try {
            int layer = std::stoi(hop_str.substr(0, colon_pos));
            int expert = std::stoi(hop_str.substr(colon_pos + 1));
            return {layer, expert};
        } catch (const std::invalid_argument& e) {
            throw ParseError("Invalid number in hop: '" + hop_str + "'. " + e.what());
        } catch (const std::out_of_range& e) {
            throw ParseError("Number out of range in hop: '" + hop_str + "'. " + e.what());
        }
    }

    std::vector<TokenState> parse_routing(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open routing file: " + filename);
        }

        std::vector<TokenState> token_states;
        std::string line;
        int token_id_counter = 0;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments

            std::stringstream ss(line);
            std::string segment;
            std::vector<std::string> parts;

            // Split line by "->"
            size_t start = 0;
            size_t arrow_pos = line.find("->");
            // Handle token ID part first: "T0: -1:-1"
            size_t first_arrow = line.find("->");
            size_t colon_pos = line.find(':');
             if (colon_pos == std::string::npos || colon_pos > first_arrow) {
                 throw ParseError("Invalid routing line format (missing initial T<id>: -1:-1): " + line);
             }
             std::string token_id_str = line.substr(1, colon_pos-1); // Extract ID assuming "T<id>" format


            std::string current_part = line.substr(colon_pos + 1); // Start after "T<id>:"
            start = 0;
            while((arrow_pos = current_part.find("->", start)) != std::string::npos) {
                 parts.push_back(current_part.substr(start, arrow_pos - start));
                 start = arrow_pos + 2; // Move past "->"
            }
            parts.push_back(current_part.substr(start)); // Add the last part

            TokenState current_token;
            try {
                 current_token.id = std::stoi(token_id_str);
             } catch(...) {
                 // If T<id> format fails, assign sequential IDs
                 std::cerr << "Warning: Could not parse token ID from '" << line.substr(0, colon_pos)
                           << "'. Assigning sequential ID " << token_id_counter << std::endl;
                 current_token.id = token_id_counter++;
             }


            if (parts.empty()) {
                 std::cerr << "Warning: Skipping empty path for token " << current_token.id << std::endl;
                continue;
            }

            // Set initial state
            current_token.current_location = parse_hop(parts[0]); // Should be -1:-1
             if(current_token.current_location.layer != -1 || current_token.current_location.expert != -1) {
                 throw ParseError("Token " + std::to_string(current_token.id) + " path must start with -1:-1");
             }

            // Parse path segments
            for (size_t i = 0; i < parts.size() - 1; ++i) {
                Hop source = parse_hop(parts[i]);
                Hop target = parse_hop(parts[i + 1]);
                current_token.path.push_back({source, target});
            }
            current_token.current_segment_index = 0;
            current_token.finished = current_token.path.empty();


            token_states.push_back(current_token);
        }
        file.close();
        return token_states;
    }

    void parse_topology(const std::string& filename,
                        ExpertDeviceMap& expert_to_device,
                        DeviceExpertMap& device_to_experts,
                        int& num_devices)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open topology file: " + filename);
        }

        expert_to_device.clear();
        device_to_experts.clear();
        num_devices = 0;
        std::string line;
        int current_device_id = 0; // Assume device IDs are sequential if not specified like "device0"

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) {
                throw ParseError("Invalid topology line format: " + line);
            }

            std::string device_name = line.substr(0, colon_pos);
            // Optional: Parse device ID from name like "deviceX"
            try {
                 if(device_name.rfind("device", 0) == 0) { // starts with device
                     current_device_id = std::stoi(device_name.substr(6)); // device0 -> 0
                 } else {
                     // If not named deviceX, assume sequential IDing based on line order
                     current_device_id = num_devices;
                 }
            } catch(...) {
                std::cerr << "Warning: Could not parse device ID from '" << device_name
                          << "'. Using sequential ID " << num_devices << std::endl;
                 current_device_id = num_devices;
            }


            std::string experts_str = line.substr(colon_pos + 1);
            std::stringstream ss(experts_str);
            std::string expert_id_str;
            std::vector<int> experts_on_device;

            while (std::getline(ss, expert_id_str, ',')) {
                 // Trim whitespace
                 expert_id_str.erase(0, expert_id_str.find_first_not_of(" \t"));
                 expert_id_str.erase(expert_id_str.find_last_not_of(" \t") + 1);

                if (expert_id_str.empty()) continue;

                try {
                    int expert_id = std::stoi(expert_id_str);
                    if (expert_to_device.count(expert_id)) {
                        throw ParseError("Duplicate expert ID found: " + std::to_string(expert_id));
                    }
                    expert_to_device[expert_id] = current_device_id;
                    experts_on_device.push_back(expert_id);
                } catch (const std::invalid_argument& e) {
                    throw ParseError("Invalid expert ID '" + expert_id_str + "' on line: " + line);
                } catch (const std::out_of_range& e) {
                    throw ParseError("Expert ID '" + expert_id_str + "' out of range on line: " + line);
                }
            }
            if(!experts_on_device.empty()) {
                device_to_experts[current_device_id] = experts_on_device;
                 num_devices = std::max(num_devices, current_device_id + 1);
            }
        }
        file.close();
         if (expert_to_device.empty()){
             throw ParseError("No expert-device mappings found in " + filename);
         }
    }

    CommunicationMatrix parse_communication(const std::string& filename,
                                            int& num_layers,
                                            int& num_experts_per_layer)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open communication file: " + filename);
        }

        CommunicationMatrix matrix;
        std::string line;
        int rows = 0;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::stringstream ss(line);
            std::vector<double> row;
            double value;
            while (ss >> value) {
                row.push_back(value);
            }

            if (!matrix.empty() && row.size() != matrix[0].size()) {
                throw ParseError("Inconsistent number of columns in communication matrix file: " + filename);
            }
            if (!row.empty()) {
                matrix.push_back(row);
                rows++;
            }
        }
        file.close();

        if (matrix.empty()) {
            throw ParseError("Communication matrix file is empty or invalid: " + filename);
        }

        int total_experts = matrix.size();
        if (total_experts != matrix[0].size()) {
             throw ParseError("Communication matrix must be square.");
        }

        // Infer L and E: Find L * E = total_experts. Requires assumptions or more info.
        // Simplest assumption: Check if total_experts is a product of two integers > 1.
        // This is ambiguous. Let's *require* the user to know L and E or provide them elsewhere.
        // For now, we cannot reliably determine L and E from the matrix size alone.
        // *** TEMPORARY/PLACEHOLDER: Assume square root? Highly likely wrong. ***
        // A better approach would be to get L from routing paths and E from topology.
        // Let's postpone L and E determination until the Simulator class where more context exists.
        num_layers = -1; // Indicate undetermined
        num_experts_per_layer = -1; // Indicate undetermined
        std::cerr << "Warning: Cannot reliably determine num_layers and num_experts_per_layer from communication matrix size alone ("
                  << total_experts << "x" << total_experts <<"). These need to be inferred or provided." << std::endl;


        return matrix;
    }

} // namespace Parser