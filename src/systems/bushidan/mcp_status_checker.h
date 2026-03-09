```cpp
#ifndef BUSHIDAN_MCP_STATUS_CHECKER_H
#define BUSHIDAN_MCP_STATUS_CHECKER_H

#include <string>
#include <vector>
#include <memory>
#include "systems/bushidan/types.h"

namespace bushidan {
namespace system {

/**
 * @brief MCP Status Checker interface for Bushidan System
 */
class MCPPairStatus {
public:
    std::string mcp_id;
    bool is_healthy;
    std::string status_message;
    int cpu_usage_percent;
    int memory_usage_mb;
    int disk_usage_percent;
    std::vector<std::string> warnings;

    MCPPairStatus(const std::string& id) : 
        mcp_id(id), 
        is_healthy(true),
        cpu_usage_percent(0),
        memory_usage_mb(0),
        disk_usage_percent(0) {}
};

class MCPStatusChecker {
public:
    /**
     * @brief Constructor for MCP Status Checker
     * @param heavy_mode Enable heavy mode processing
     */
    explicit MCPStatusChecker(bool heavy_mode = false);
    
    /**
     * @brief Destructor
     */
    ~MCPStatusChecker() = default;

    /**
     * @brief Check status of all MCP pairs in system
     * @return Vector of MCPPairStatus objects containing health info
     */
    std::vector<MCPPairStatus> check_all_mcp_status();

    /**
     * @brief Check status for specific MCP pair
     * @param mcp_id ID of the MCP to check
     * @return Status information for the specified MCP
     */
    MCPPairStatus check_single_mcp(const std::string& mcp_id);

    /**
     * @brief Get system configuration parameters
     * @return Configuration object with checking rules and thresholds
     */
    const Config& get_config() const;

private:
    bool heavy_mode_;
    std::unique_ptr<Config> config_;

    /**
     * @brief Load configuration from file paths
     * @param config_path Path to JSON config file
     * @param rules_path Path to YAML rule file
     * @return True if successful, false otherwise
     */
    bool load_configuration(const std::string& config_path, const std::string& rules_path);

    /**
     * @brief Validate MCP health based on configured thresholds
     * @param status Status object to validate
     * @return True if healthy, false otherwise
     */
    bool validate_health(const MCPPairStatus& status) const;

    /**
     * @brief Get list of active MCP pairs from Bushidan system
     * @return Vector of MCP IDs
     */
    std::vector<std::string> get_active_mcp_list() const;
};

} // namespace system
} // namespace bushidan

#endif // BUSHIDAN_MCP_STATUS_CHECKER_H
```