#pragma once
#include "types.h"
#include "ring_buffer.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <random>
#include <functional>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────
//  ModelSimulator
//
//  In a real deployment this class would be replaced (or wrapped) by
//  a hook into llama.cpp / ggml callbacks.  The interface is kept
//  identical so swapping is a one-line change.
//
//  It runs a background thread that generates realistic-looking
//  LayerPackets and pushes them into a shared ring-buffer.
// ─────────────────────────────────────────────────────────────────

constexpr std::size_t PACKET_RING_SIZE = 256;

struct SimState {
    RingBuffer<LayerPacket, PACKET_RING_SIZE> packets;
    std::vector<AnomalyEntry>                 anomalies;
    TopoNode                                  topo_root;
    uint64_t                                  total_packets = 0;
    std::mutex                                mtx;
};

class ModelSimulator {
public:
    explicit ModelSimulator(SimState& state) : state_(state), running_(false), rng_(42) {
        build_topology();
        build_token_list();
    }

    void start() {
        running_ = true;
        worker_ = std::thread([this]{ run(); });
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

    ~ModelSimulator() { stop(); }

private:
    // ── Topology ──────────────────────────────────────────────────
    void build_topology() {
        auto& root = state_.topo_root;
        root.name  = "llama-3-8b";
        root.type  = LayerType::Unknown;
        root.depth = 0;
        root.expanded = true;

        // embed_tokens
        TopoNode embed;
        embed.name = "embed_tokens"; embed.type = LayerType::Embedding;
        embed.depth = 1; embed.expanded = false;
        root.children.push_back(embed);

        // layers 0..N_LAYERS-1
        TopoNode layers_node;
        layers_node.name = "layers"; layers_node.depth = 1;
        layers_node.type = LayerType::Unknown; layers_node.expanded = true;

        for (int i = 0; i < N_LAYERS; ++i) {
            TopoNode ln;
            ln.name = "layers." + std::to_string(i);
            ln.depth = 2; ln.type = LayerType::Unknown; ln.expanded = false;

            TopoNode attn; attn.name = ln.name + ".self_attn";
            attn.type = LayerType::SelfAttention; attn.depth = 3;
            TopoNode mlp;  mlp.name  = ln.name + ".mlp";
            mlp.type  = LayerType::MLP; mlp.depth = 3;
            TopoNode norm1; norm1.name = ln.name + ".input_layernorm";
            norm1.type = LayerType::RMSNorm; norm1.depth = 3;
            TopoNode norm2; norm2.name = ln.name + ".post_attn_layernorm";
            norm2.type = LayerType::RMSNorm; norm2.depth = 3;

            ln.children = {norm1, attn, norm2, mlp};
            layers_node.children.push_back(ln);
        }
        root.children.push_back(layers_node);

        TopoNode norm_f; norm_f.name = "norm"; norm_f.type = LayerType::RMSNorm; norm_f.depth = 1;
        TopoNode lm_head; lm_head.name = "lm_head"; lm_head.type = LayerType::Output; lm_head.depth = 1;
        root.children.push_back(norm_f);
        root.children.push_back(lm_head);
    }

    void build_token_list() {
        token_seq_ = {"I", "want", "it", "to", "be", "keyboard", "driven",
                       "and", "fast", "at", "runtime", "with", "low", "memory"};
    }

    // ── Background simulation thread ──────────────────────────────
    void run() {
        // Sequence of layers in forward pass order
        std::vector<std::pair<std::string,LayerType>> seq;
        seq.push_back({"embed_tokens", LayerType::Embedding});
        for (int i = 0; i < N_LAYERS; ++i) {
            std::string p = "layers." + std::to_string(i);
            seq.push_back({p + ".input_layernorm",      LayerType::RMSNorm});
            seq.push_back({p + ".self_attn",            LayerType::SelfAttention});
            seq.push_back({p + ".post_attn_layernorm",  LayerType::RMSNorm});
            seq.push_back({p + ".mlp",                  LayerType::MLP});
        }
        seq.push_back({"norm",    LayerType::RMSNorm});
        seq.push_back({"lm_head", LayerType::Output});

        int step = 0;
        while (running_) {
            auto& [name, type] = seq[step % seq.size()];
            LayerPacket pkt = generate_packet(name, type, step);

            {
                std::lock_guard<std::mutex> lk(state_.mtx);
                state_.packets.push(pkt);
                state_.total_packets++;
                if (pkt.is_anomaly) {
                    AnomalyEntry ae;
                    ae.timestamp = pkt.timestamp;
                    ae.severity  = AnomalySeverity::WARN;
                    ae.message   = pkt.anomaly_msg;
                    if (state_.anomalies.size() > 200)
                        state_.anomalies.erase(state_.anomalies.begin());
                    state_.anomalies.push_back(ae);
                }
            }

            ++step;
            // Simulate realistic per-layer timing
            int sleep_ms = base_latency(type);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }

    int base_latency(LayerType t) {
        switch (t) {
            case LayerType::SelfAttention: return 80 + uniform(0,20);
            case LayerType::MLP:           return 60 + uniform(0,15);
            case LayerType::RMSNorm:       return 8  + uniform(0,4);
            case LayerType::LayerNorm:     return 10 + uniform(0,5);
            case LayerType::Embedding:     return 20 + uniform(0,5);
            case LayerType::Output:        return 15 + uniform(0,5);
            default:                       return 10;
        }
    }

    LayerPacket generate_packet(const std::string& name, LayerType type, int /*step*/) {
        LayerPacket p;
        p.packet_id  = state_.total_packets + 1;
        p.timestamp  = now_str();
        p.layer_name = name;
        p.layer_type = type;

        // Mostly GPU, occasionally CPU fallback
        if (uniform(0,100) < 8)
            p.device = ComputeDevice::CPU;
        else
            p.device = (uniform(0,2) == 0) ? ComputeDevice::CUDA_GPU1 : ComputeDevice::CUDA_GPU0;

        // Latency (ms) with some noise
        double base = (double)base_latency(type);
        p.latency_ms = base * (0.9 + 0.2 * uniform_f());

        // Tensor shapes
        int seq_len = (int)token_seq_.size();
        switch (type) {
            case LayerType::Embedding:
                p.output_shape = {{1, seq_len, 4096}, "float16"};
                break;
            case LayerType::SelfAttention:
                p.output_shape = {{1, N_HEADS, seq_len, seq_len}, "float16"};
                break;
            case LayerType::MLP:
                p.output_shape = {{1, seq_len, 4096}, "float16"};
                break;
            case LayerType::RMSNorm:
            case LayerType::LayerNorm:
                p.output_shape = {{1, seq_len, 4096}, "float16"};
                break;
            case LayerType::Output:
                p.output_shape = {{1, seq_len, 32000}, "float32"};
                break;
            default:
                p.output_shape = {{1, seq_len, 4096}, "float16"};
        }

        // Activation stats
        p.activation_mean = 0.01f + 0.05f * uniform_f();
        p.activation_max  = 1.5f  + 5.0f  * uniform_f();
        p.activation_min  = -p.activation_max * (0.8f + 0.4f * uniform_f());
        p.sparsity_pct    = 30.0 + 40.0 * uniform_f();

        // Anomaly detection
        float anomaly_roll = uniform_f();
        if (p.activation_max > 6.0f || anomaly_roll < 0.04f) {
            p.is_anomaly = true;
            if (p.device == ComputeDevice::CPU)
                p.anomaly_msg = "CUDA OOM Fallback: Processing " + name + " on CPU Host Memory.";
            else if (p.activation_max > 6.0f)
                p.anomaly_msg = "Outlier Feature " + name + ": Max=" + fmtf(p.activation_max) + " > 6.0";
            else
                p.anomaly_msg = "High sparsity in " + name + ": " + fmtf(p.sparsity_pct) + "%";
        }

        // Attention matrix for attention layers
        if (type == LayerType::SelfAttention) {
            p.has_attn = true;
            p.attn = generate_attention(seq_len);
        }

        return p;
    }

    AttentionMatrix generate_attention(int n) {
        int real_n = std::min(n, MAX_ATTN_TOKENS);
        AttentionMatrix m;
        m.num_tokens = real_n;
        m.token_labels = std::vector<std::string>(token_seq_.begin(),
                                                   token_seq_.begin() + real_n);
        m.num_heads  = N_HEADS;
        m.active_head = 0;

        // Generate N_HEADS attention matrices with causal masking + softmax
        m.weights.resize(N_HEADS);
        for (int h = 0; h < N_HEADS; ++h) {
            m.weights[h].resize(real_n, std::vector<float>(real_n, 0.f));
            for (int r = 0; r < real_n; ++r) {
                // Raw logits (causal — upper triangle = -inf)
                float sum = 0.f;
                std::vector<float> row(real_n, 0.f);
                for (int c = 0; c <= r; ++c) {
                    float v = uniform_f();
                    // Diagonal boost to get the realistic "attending to self" pattern
                    if (c == r) v += 0.5f + uniform_f();
                    else if (r - c < 3) v += 0.2f * uniform_f();
                    row[c] = v;
                    sum += v;
                }
                for (int c = 0; c <= r; ++c)
                    m.weights[h][r][c] = row[c] / sum;
            }
        }
        return m;
    }

    // ── Utilities ─────────────────────────────────────────────────
    int uniform(int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(rng_);
    }
    float uniform_f() {
        return std::uniform_real_distribution<float>(0.f, 1.f)(rng_);
    }
    std::string fmtf(float v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << v;
        return oss.str();
    }
    std::string now_str() {
        auto now  = std::chrono::system_clock::now();
        auto t    = std::chrono::system_clock::to_time_t(now);
        auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) % 1000;
        std::tm tm_s{}; localtime_r(&t, &tm_s);
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
                      tm_s.tm_hour, tm_s.tm_min, tm_s.tm_sec, (int)ms.count());
        return buf;
    }

    static constexpr int N_LAYERS = 32;
    static constexpr int N_HEADS  = 32;

    SimState&             state_;
    std::atomic<bool>     running_;
    std::thread           worker_;
    std::mt19937          rng_;
    std::vector<std::string> token_seq_;
};
