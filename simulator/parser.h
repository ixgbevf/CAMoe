#ifndef PARSER_H
#define PARSER_H

#include "data_structures.h"
#include <string>
#include <vector>
#include <stdexcept>

namespace Parser {

    // Exception class for parsing errors
    class ParseError : public std::runtime_error {
    public:
        ParseError(const std::string& message) : std::runtime_error(message) {}
    };

    // Parses routing.txt into a vector of TokenState objects
    std::vector<TokenState> parse_routing(const std::string& filename);

    // Parses topology.txt into ExpertDeviceMap and DeviceExpertMap
    void parse_topology(const std::string& filename,
                        ExpertDeviceMap& expert_to_device,
                        DeviceExpertMap& device_to_experts,
                        int& num_devices);

    // Parses communication.txt into the bandwidth matrix
    // Also determines number of layers and experts per layer based on matrix size
    CommunicationMatrix parse_communication(const std::string& filename,
                                            int& num_layers,
                                            int& num_experts_per_layer);

} // namespace Parser

#endif // PARSER_H