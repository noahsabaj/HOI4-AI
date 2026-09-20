#pragma once
#include <nlohmann/json.hpp>

namespace arena {
// Implement only after the corresponding HOI4 ABI and thread context are verified.
// This interface does not establish that any game function has been resolved.
class EngineAdapter {
public:
    virtual ~EngineAdapter() = default;
    virtual nlohmann::json capabilities() const = 0;
    virtual nlohmann::json fingerprint() const = 0;
    // Implementations must marshal to the simulation thread and produce a
    // player-specific copy. The pipe thread must never dereference live units.
    virtual nlohmann::json reset(const nlohmann::json& spec) = 0;
    virtual nlohmann::json observe(const nlohmann::json& player) = 0;
    virtual nlohmann::json submit(const nlohmann::json& order) = 0;
};
}
