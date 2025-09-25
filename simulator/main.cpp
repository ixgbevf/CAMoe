#include <systemc.h>
#include "simulator.h"
#include "config.h" // Include for time unit configuration

int sc_main(int argc, char* argv[]) {
    // Set simulation time resolution globally if needed
     sc_core::sc_set_time_resolution(1, sc_core::SC_PS); // Example: 1 picosecond resolution

    Simulator moe_sim("MoE_Simulator");

    std::cout << "Starting SystemC simulation..." << std::endl;
    sc_core::sc_start(); // Run the simulation until sc_stop() or naturally ends
    std::cout << "SystemC simulation ended." << std::endl;

    return 0;
}