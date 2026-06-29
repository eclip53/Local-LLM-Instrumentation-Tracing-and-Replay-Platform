#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <array>

// ─────────────────────────────────────────────
//  Enumerations
// ─────────────────────────────────────────────

enum class LayerType {
    Embedding,
    SelfAttention,
    MLP,
    LayerNorm,
    RMSNorm,
    Output,
    Unknown
};

enum class ComputeDevice {
    CPU,
    CUDA_GPU0,
    CUDA_GPU1,
    METAL,
    Unknown
};

// ─────────────────────────────────────────────
//  Attention Data
// ─────────────────────────────────────────────

constexpr int MAX_ATTN_TOKENS = 16;

struct AttentionMatrix {
    int num_tokens = 0;
    std::vector<std::string> token_labels;
    // attention weights [head][row][col]
    std::vector<std::vector<std::vector<float>>> weights;
    int num_heads = 0;
    int active_head = 0;
};

// ─────────────────────────────────────────────
//  Tensor Metadata
// ─────────────────────────────────────────────

struct TensorShape {
    std::vector<int64_t> dims;
    std::string dtype; // "float32", "float16", "bfloat16", "int8"

    std::string to_string() const {
        std::string s = "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            s += std::to_string(dims[i]);
            if (i + 1 < dims.size()) s += ", ";
        }
        return s + "]";
    }
};

// ─────────────────────────────────────────────
//  Anomaly / Event Log Entry
// ─────────────────────────────────────────────

enum class AnomalySeverity { INFO, WARN, ERROR };

struct AnomalyEntry {
    std::string timestamp;
    AnomalySeverity severity;
    std::string message;
};

// ─────────────────────────────────────────────
//  Layer Packet — one captured forward-pass record
// ─────────────────────────────────────────────

struct LayerPacket {
    uint64_t packet_id = 0;
    std::string timestamp;
    std::string layer_name;
    LayerType   layer_type = LayerType::Unknown;
    ComputeDevice device   = ComputeDevice::CPU;

    // Runtime metrics
    double latency_ms    = 0.0;
    double sparsity_pct  = 0.0;
    float  activation_mean = 0.f;
    float  activation_max  = 0.f;
    float  activation_min  = 0.f;
    TensorShape output_shape;

    // Only populated for attention layers
    AttentionMatrix attn;
    bool has_attn = false;

    // Anomaly flags
    bool is_anomaly     = false;
    std::string anomaly_msg;
};

// ─────────────────────────────────────────────
//  Model Topology node
// ─────────────────────────────────────────────

struct TopoNode {
    std::string name;
    LayerType   type = LayerType::Unknown;
    bool        expanded = false;
    int         depth    = 0;
    std::vector<TopoNode> children;
};

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

inline std::string layer_type_str(LayerType t) {
    switch (t) {
        case LayerType::Embedding:     return "Embedding";
        case LayerType::SelfAttention: return "Attn (Self)";
        case LayerType::MLP:           return "MLP (SwiGLU)";
        case LayerType::LayerNorm:     return "LayerNorm";
        case LayerType::RMSNorm:       return "RMSNorm";
        case LayerType::Output:        return "Output (LM Head)";
        default:                       return "Unknown";
    }
}

inline std::string device_str(ComputeDevice d) {
    switch (d) {
        case ComputeDevice::CUDA_GPU0: return "CUDA [GPU 0]";
        case ComputeDevice::CUDA_GPU1: return "CUDA [GPU 1]";
        case ComputeDevice::METAL:     return "Metal (Apple)";
        case ComputeDevice::CPU:       return "CPU (Fallback)";
        default:                       return "Unknown";
    }
}
