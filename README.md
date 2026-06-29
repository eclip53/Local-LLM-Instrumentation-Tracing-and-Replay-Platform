# LLM-Tracer — Local LLM Instrumentation, Tracing, and Replay Platform

A lightweight **C++17** telemetry and diagnostic TUI for local transformer models.  
Inspired by `btop` / `lazygit`; built with ncurses.

```
 LLM-Tracer v1.0  |  Model: llama-3-8b     [Tab] Cycle Focus  [j/k] Nav  [H/L] Head  [+/-] Contrast  [q] Quit

╔═ █ 1. MODEL TOPOLOGY (Focus Active) ══╗  ╔═ 2. LIVE PACKET STREAM ════════════════════════════════╗
║ ▼ llama-3-8b                          ║  ║ ID     TIMESTAMP      LAYER TYPE       DEVICE           ║
║   ► embed_tokens                      ║  ║ 104    21:14:02.110   Attn (Self)      CUDA [GPU 0]     ║
║   ▼ layers                            ║  ║ 105    21:14:02.114   MLP (SwiGLU)     CUDA [GPU 0]     ║
║     ▼ layers.1  ● self_attn           ║  ║ 106    21:14:02.119   RMSNorm          CPU (Fallback)   ║
║                 ● mlp                 ║  ╚════════════════════════════════════════════════════════╝
╚═══════════════════════════════════════╝

╔═ 3. ATTENTION MATRIX VISUALIZER (Head 0/31) ══════════════════════════════════════════════════════╗
║ Tokens:  [I]    [want]  [it]   [to]   [be]   [keyboard] [driven]         Viewport: [0-7] x [0-7] ║
║ [I]      ████   ░░░░    ░░░░   ░░░░   ░░░░    ░░░░       ░░░░                                    ║
║ [want]   ▒▒▒▒   ████    ░░░░   ░░░░   ░░░░    ░░░░       ░░░░                                    ║
║ [it]     ░░░░   ▒▒▒▒    ████   ░░░░   ░░░░    ░░░░       ░░░░                                    ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════╝

╔═ 4. RUNTIME METRICS INSPECTOR ════════╗  ╔═ 5. NUMERICAL ANOMALY LEDGER ═════════════════════════╗
║ Layer  : layers.1.self_attn           ║  ║ 21:14:02.114 ⚠ Outlier Feature Layer 0: Max > 6.0    ║
║ Tensor : [1, 32, 4096]  Dtype: f16   ║  ║ 21:14:02.128 ✖ CUDA OOM Fallback: CPU Host Memory.   ║
║ Latency: 1.142 ms  ✓ Normal          ║  ╚════════════════════════════════════════════════════════╝
╚═══════════════════════════════════════╝
```

---

## Features

| Feature |
|---|---|
| 5-panel interactive TUI |
| Vim-style `j/k` navigation |
| `Tab` to cycle panel focus | 
| Model topology tree (expand/collapse with Space) |
| Live packet stream with per-layer latency |
| Causal attention matrix heatmap |
| Multi-head navigation (`H`/`L`) |
| Attention viewport panning (`h`/`j`/`k`/`l`) |
| Contrast adjustment (`+`/`-`) | 
| Runtime metrics (shape, dtype, sparsity, latency) | 
| Numerical anomaly ledger (outlier/OOM detection) | 
| Fixed-size ring buffer (no RAM blowup) | 
| Non-invasive hook API for real llama.cpp |
| Simulator for offline demo/testing | 
| Terminal resize handling | 

---

## Build

### Prerequisites
```bash
# Ubuntu / Debian
sudo apt install libncursesw5-dev g++

# Arch
sudo pacman -S ncurses gcc
```

### Compile
```bash
make          # builds ./llm-tracer
make run      # builds and launches immediately
make clean    # remove binary
```

Requires **C++17**, `g++ ≥ 8`, and `libncursesw`.

---

## Keyboard Reference

| Key | Context | Action |
|---|---|---|
| `Tab` | Global | Cycle panel focus |
| `q` / `Q` | Global | Quit |
| `j` / `↓` | Topology | Move selection down |
| `k` / `↑` | Topology | Move selection up |
| `Space` | Topology | Expand / Collapse node |
| `j` / `k` | Stream | Scroll packet history |
| `h`/`j`/`k`/`l` | Attention | Pan viewport |
| `H` / `L` | Attention | Previous / Next attention head |
| `+` / `-` | Attention | Increase / Decrease weight contrast |

---

## Architecture

```
include/
  types.h        — Core data structures (LayerPacket, AttentionMatrix, TopoNode, …)
  ring_buffer.h  — Fixed-size ring buffer (template, N=256 packets)
  simulator.h    — Background thread simulating a 32-layer LLaMA 3 forward pass
  hook.h         — Non-invasive hook API to plug into real llama.cpp / ggml
  tui.h          — Full ncurses TUI (5 panels, input handling, rendering)
src/
  main.cpp       — Entry point; wires SimState → Simulator → TUI event loop
```

### Data flow

```
[Model / Simulator]
       │  push LayerPacket
       ▼
  SimState.packets  (RingBuffer<LayerPacket, 256>)  ← mutex-protected
  SimState.anomalies (vector, max 200 entries)
       │  snapshot() each frame
       ▼
  TUI::render()  →  5 ncurses windows  →  terminal
```

---

## Plugging into llama.cpp (real model)

Replace `ModelSimulator` with `LLMHook` from `include/hook.h`:

```cpp
#include "hook.h"

SimState g_state;
LLMHook  g_hook(g_state);

// In your llama.cpp host, register the ggml compute callback:
struct ggml_cplan cplan = ggml_graph_plan(graph, n_threads);
// (or use the newer ggml_backend_sched callback on llama.cpp main branch)
cplan.abort_callback_data = &g_hook;

// Wrap each tensor compute:
ggml_set_output_callback(ctx, [](ggml_tensor* t, void* ud) {
    static_cast<LLMHook*>(ud)->on_tensor_done(t);
}, &g_hook);
```

The hook:
- Infers `LayerType` from tensor name patterns (`attn`, `mlp`, `norm`, …)
- Infers device from `ggml_tensor::extra` (non-null ⟹ CUDA backend)
- Computes activation stats (mean, max, min, sparsity) on CPU tensors
- Flags anomalies (`max > 6.0`, OOM fallbacks)
- Pushes `LayerPacket` into `SimState` under a mutex

---

## Extension Ideas

- **Replay mode**: serialize `LayerPacket` stream to disk with `cereal` / `msgpack` and replay offline
- **GPU activation capture**: use `cudaMemcpy` in the hook for CUDA tensors
- **SVI / attention entropy**: add per-head entropy metric to `AttentionMatrix`
- **Latency flamegraph**: accumulate `latency_ms` per layer name for a bar chart view
- **llama.cpp `ggml_backend` integration**: subscribe to `ggml_backend_sched_set_eval_callback`
