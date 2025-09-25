#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <vector>
#include <map>
#include <string>

// Represents a hop target: Layer:Expert
struct Hop {
    int layer = -1;
    int expert = -1;

    bool operator==(const Hop& other) const {
        return layer == other.layer && expert == other.expert;
    }
     // For use in maps if needed
    bool operator<(const Hop& other) const {
        if (layer != other.layer) return layer < other.layer;
        return expert < other.expert;
    }
};

// Represents a single step in the path: Source Hop -> Target Hop
struct PathSegment {
    Hop source;
    Hop target;
};

// Stores the full path for a token and its current state
struct TokenState {
    int id = -1;
    std::vector<PathSegment> path;
    size_t current_segment_index = 0; // Index in the 'path' vector
    Hop current_location;             // Where the token currently resides
    bool finished = false;
};

// --- Data derived from input files ---

// Maps global expert ID to device ID
using ExpertDeviceMap = std::map<int, int>;

// Maps device ID to the list of global expert IDs it hosts
using DeviceExpertMap = std::map<int, std::vector<int>>;

// Represents the communication bandwidth matrix (Expert Instance to Expert Instance)
// Indexed by [global_source_expert_id][global_target_expert_id]
using CommunicationMatrix = std::vector<std::vector<double>>;

#endif // DATA_STRUCTURES_H