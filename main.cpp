#include <chrono>
#include <thread>
#include <csignal>

#include "types.h"
#include "ring_buffer.h"
#include "simulator.h"
#include "tui.h"

// ─────────────────────────────────────────────────────────────────
//  Global signal flag (Ctrl-C safety)
// ─────────────────────────────────────────────────────────────────
static volatile bool g_quit = false;
void sig_handler(int) { g_quit = true; }

int main() {
    std::signal(SIGINT,  sig_handler);
    std::signal(SIGTERM, sig_handler);

    // Shared state between simulator and TUI
    SimState state;

    // Start the simulator (background thread)
    ModelSimulator sim(state);
    sim.start();

    // Give the simulator a moment to fill the buffer before first render
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Build TUI (initialises ncurses inside constructor)
    TUI tui(state);

    // ── Main event loop ───────────────────────────────────────────
    constexpr int FRAME_MS = 80;  // ~12 fps — gentle on the terminal

    while (!g_quit) {
        if (!tui.handle_input()) break;   // 'q' pressed
        tui.render();
        std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_MS));
    }

    // TUI destructor calls endwin(); simulator destructor joins thread.
    return 0;
}
