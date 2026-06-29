#pragma once
// ─────────────────────────────────────────────────────────────────
//  LLM-Tracer  —  Hook Interface
//
//  This header defines the non-invasive callback API that lets you
//  plug the tracer into an actual llama.cpp / ggml runtime WITHOUT
//  modifying the model source.
//
//  Usage pattern (in your host application):
//
//    #include "hook.h"
//    #include "simulator.h"   // for SimState
//
//    SimState g_state;
//    LLMHook  g_hook(g_state);
//
//    // Register with ggml (pseudo-code — actual API varies by version):
//    ggml_set_compute_callback(ggml_ctx, [](ggml_tensor* t, bool before, void* ud){
//        auto* hook = static_cast<LLMHook*>(ud);
//        if (!before) hook->on_tensor_done(t);
//    }, &g_hook);
//
//  The hook converts ggml_tensor metadata into LayerPackets and
//  pushes them into SimState — same interface the TUI reads from.
// ─────────────────────────────────────────────────────────────────

#include "types.h"
#include "simulator.h"
#include <string>
#include <cstring>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <mutex>

// ─────────────────────────────────────────────────────────────────
//  Minimal ggml_tensor shadow  (replace with real ggml.h include
//  when building against llama.cpp)
// ─────────────────────────────────────────────────────────────────
#ifndef GGML_TYPE_F16
struct ggml_tensor {
    // ggml type enum (simplified)
    int type;           // 1=F32  2=F16  3=Q4_0  ...
    int n_dims;
    int64_t ne[4];      // dimensions
    size_t  nb[4];      // strides (bytes)
    float*  data;       // raw pointer (may be on GPU)
    char    name[64];
    void*   extra;      // backend-specific
};
#endif

// ─────────────────────────────────────────────────────────────────
//  LLMHook  —  converts ggml events → LayerPackets
// ─────────────────────────────────────────────────────────────────
class LLMHook {
public:
    explicit LLMHook(SimState& state) : state_(state) {}

    // Call this from your ggml compute callback (after=true)
    void on_tensor_done(const ggml_tensor* t) {
        if (!t || t->name[0] == '\0') return;

        LayerPacket pkt;
        pkt.packet_id  = next_id_++;
        pkt.timestamp  = now_str();
        pkt.layer_name = t->name;
        pkt.layer_type = infer_type(t->name);
        pkt.device     = infer_device(t);

        // Tensor shape
        pkt.output_shape.dims.clear();
        for (int d = 0; d < t->n_dims; ++d)
            pkt.output_shape.dims.push_back(t->ne[d]);
        pkt.output_shape.dtype = dtype_str(t->type);

        // Activation stats (CPU tensors only — GPU tensors need a copy)
        if (t->data && pkt.device == ComputeDevice::CPU) {
            compute_stats(t, pkt);
        } else {
            pkt.activation_mean = 0.f;
            pkt.activation_max  = 0.f;
            pkt.activation_min  = 0.f;
            pkt.sparsity_pct    = 0.0;
        }

        // Anomaly check
        if (pkt.activation_max > 6.0f) {
            pkt.is_anomaly = true;
            pkt.anomaly_msg = "Outlier Feature " + pkt.layer_name +
                              ": Max=" + fmt(pkt.activation_max) + " > 6.0";
        }

        // Latency: measured externally with ggml_time_us() if available
        // Set to 0 here; host can fill via set_last_latency()
        pkt.latency_ms = last_latency_ms_;

        {
            std::lock_guard<std::mutex> lk(state_.mtx);
            state_.packets.push(pkt);
            state_.total_packets++;
            if (pkt.is_anomaly) {
                AnomalyEntry ae;
                ae.timestamp = pkt.timestamp;
                ae.severity  = AnomalySeverity::WARN;
                ae.message   = pkt.anomaly_msg;
                state_.anomalies.push_back(ae);
            }
        }
    }

    // Optional: set latency from ggml_time_us() delta
    void set_last_latency(double ms) { last_latency_ms_ = ms; }

private:
    // ── Helpers ───────────────────────────────────────────────────
    static LayerType infer_type(const char* name) {
        std::string n(name);
        if (n.find("attn")   != std::string::npos ||
            n.find("self_at") != std::string::npos) return LayerType::SelfAttention;
        if (n.find("mlp")    != std::string::npos) return LayerType::MLP;
        if (n.find("norm")   != std::string::npos) return LayerType::RMSNorm;
        if (n.find("embed")  != std::string::npos) return LayerType::Embedding;
        if (n.find("lm_head")!= std::string::npos) return LayerType::Output;
        return LayerType::Unknown;
    }

    static ComputeDevice infer_device(const ggml_tensor* t) {
        // ggml sets t->extra to a CUDA buffer when using CUDA backend
        if (t->extra != nullptr)  return ComputeDevice::CUDA_GPU0;
        return ComputeDevice::CPU;
    }

    static std::string dtype_str(int type) {
        switch (type) {
            case 0: return "f32";
            case 1: return "f16";
            case 2: return "q4_0";
            case 3: return "q4_1";
            case 6: return "q8_0";
            default: return "unk";
        }
    }

    static void compute_stats(const ggml_tensor* t, LayerPacket& pkt) {
        // Only works for float tensors on CPU
        if (t->type != 0 && t->type != 1) return;
        const float* data = reinterpret_cast<const float*>(t->data);
        int64_t total = 1;
        for (int d = 0; d < t->n_dims; ++d) total *= t->ne[d];
        total = std::min(total, (int64_t)8192); // cap sample

        double sum = 0.0; float mn = 1e9f, mx = -1e9f; int64_t zeros = 0;
        for (int64_t i = 0; i < total; ++i) {
            float v = data[i];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (std::abs(v) < 1e-6f) ++zeros;
        }
        pkt.activation_mean = (float)(sum / total);
        pkt.activation_min  = mn;
        pkt.activation_max  = mx;
        pkt.sparsity_pct    = 100.0 * zeros / total;
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

    static std::string fmt(float v) {
        std::ostringstream o; o << std::fixed << std::setprecision(2) << v;
        return o.str();
    }

    SimState& state_;
    uint64_t  next_id_       = 1;
    double    last_latency_ms_ = 0.0;
};
