#ifndef CONFIG_H
#define CONFIG_H

#include <systemc.h>

namespace config {
    // User-configurable Parameters (Example Values)
    const double LINK_LATENCY_L = 10.0;        // ns (Fixed link latency)
    const double OVERHEAD_O = 5.0;             // ns (Protocol processing/scheduling)
    const double DATA_SIZE_N = 1024.0;         // bytes (Per token)
    const double MAX_PAYLOAD = 256.0;          // bytes (Max payload per packet)
    const double FLIT_SIZE = 64.0;             // bytes (Header overhead per packet)
    const double INTRA_DEVICE_LATENCY = 2.0; // ns (Latency for expert comms on same device)
    const bool ENABLE_INTRA_DEVICE_LATENCY = true; // Consider intra-device latency?
    const bool ENABLE_DATA_AGGREGATION = true;    // Aggregate data for same target device?

    // Input file paths
    const std::string ROUTING_FILE = "routing.txt";
    const std::string TOPOLOGY_FILE = "topology.txt";
    const std::string COMMUNICATION_FILE = "communication.txt"; // Assumed to be Bandwidth (Bytes/ns)

    // Simulation Defaults
    const sc_core::sc_time_unit TIME_UNIT = sc_core::SC_NS;
} // namespace config

#endif // CONFIG_H