#pragma once
#include <ncurses.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "types.h"
#include "ring_buffer.h"
#include "simulator.h"

// ─────────────────────────────────────────────────────────────────
//  Color pair IDs
// ─────────────────────────────────────────────────────────────────
enum Color : int {
    C_NORMAL   = 1,
    C_HEADER   = 2,
    C_FOCUSED  = 3,
    C_SELECTED = 4,
    C_WARN     = 5,
    C_ERROR    = 6,
    C_INFO     = 7,
    C_CUDA     = 8,
    C_CPU      = 9,
    C_DIM      = 10,
    C_TITLE    = 11,
    C_ATTN_HI  = 12,
    C_ATTN_MED = 13,
    C_ATTN_LO  = 14,
    C_GREEN    = 15,
    C_BORDER_F = 16,  // focused border
};

// ─────────────────────────────────────────────────────────────────
//  Panel focus enum
// ─────────────────────────────────────────────────────────────────
enum class Panel { Topology, Stream, Attention, Metrics, Anomalies };

// ─────────────────────────────────────────────────────────────────
//  TUI — manages all ncurses windows and rendering
// ─────────────────────────────────────────────────────────────────
class TUI {
public:
    TUI(SimState& state) : state_(state), focus_(Panel::Topology) {
        initscr();
        cbreak();
        noecho();
        keypad(stdscr, TRUE);
        nodelay(stdscr, TRUE);
        curs_set(0);
        start_color();
        use_default_colors();
        setup_colors();
        get_dims();
        create_windows();
    }

    ~TUI() {
        destroy_windows();
        endwin();
    }

    // Returns false when user presses Q
    bool handle_input() {
        int ch = getch();
        if (ch == ERR) return true; // no input

        switch (ch) {
            case '\t':  // Tab — cycle focus
                cycle_focus();
                break;
            case 'q': case 'Q':
                return false;
            case 'j': case KEY_DOWN:
                on_down();
                break;
            case 'k': case KEY_UP:
                on_up();
                break;
            case 'h': case KEY_LEFT:
                on_left();
                break;
            case 'l': case KEY_RIGHT:
                on_right();
                break;
            case ' ':
                on_space();
                break;
            case '+': case '=':
                attn_contrast_ = std::min(3.0f, attn_contrast_ + 0.2f);
                break;
            case '-':
                attn_contrast_ = std::max(0.2f, attn_contrast_ - 0.2f);
                break;
            case 'H': case '<':  // previous attention head
                if (sel_pkt_ && sel_pkt_->has_attn)
                    sel_pkt_->attn.active_head = std::max(0, sel_pkt_->attn.active_head - 1);
                break;
            case 'L': case '>':  // next attention head
                if (sel_pkt_ && sel_pkt_->has_attn)
                    sel_pkt_->attn.active_head = std::min(
                        sel_pkt_->attn.num_heads - 1, sel_pkt_->attn.active_head + 1);
                break;
            case KEY_RESIZE:
                destroy_windows();
                get_dims();
                create_windows();
                break;
        }
        return true;
    }

    void render() {
        // Snapshot shared state under lock
        std::vector<LayerPacket>  pkts;
        std::vector<AnomalyEntry> anoms;
        {
            std::lock_guard<std::mutex> lk(state_.mtx);
            pkts  = state_.packets.snapshot();
            anoms = state_.anomalies;
        }

        // Auto-select latest packet for metrics/attention
        if (!pkts.empty()) {
            latest_pkt_ = pkts.back();
            if (!sel_pkt_locked_) {
                sel_pkt_ = &latest_pkt_;
            }
        }

        werase(stdscr);
        render_statusbar();
        render_topology();
        render_stream(pkts);
        render_attention();
        render_metrics();
        render_anomalies(anoms);
        doupdate();
    }

private:
    // ── Color setup ───────────────────────────────────────────────
    void setup_colors() {
        init_pair(C_NORMAL,   COLOR_WHITE,   -1);
        init_pair(C_HEADER,   COLOR_BLACK,   COLOR_CYAN);
        init_pair(C_FOCUSED,  COLOR_CYAN,    -1);
        init_pair(C_SELECTED, COLOR_BLACK,   COLOR_GREEN);
        init_pair(C_WARN,     COLOR_YELLOW,  -1);
        init_pair(C_ERROR,    COLOR_RED,     -1);
        init_pair(C_INFO,     COLOR_CYAN,    -1);
        init_pair(C_CUDA,     COLOR_GREEN,   -1);
        init_pair(C_CPU,      COLOR_RED,     -1);
        init_pair(C_DIM,      COLOR_BLACK,   -1);
        init_pair(C_TITLE,    COLOR_MAGENTA, -1);
        init_pair(C_ATTN_HI,  COLOR_WHITE,   COLOR_RED);
        init_pair(C_ATTN_MED, COLOR_WHITE,   COLOR_YELLOW);
        init_pair(C_ATTN_LO,  COLOR_WHITE,   COLOR_BLACK);
        init_pair(C_GREEN,    COLOR_GREEN,   -1);
        init_pair(C_BORDER_F, COLOR_CYAN,    -1);
    }

    // ── Window geometry ───────────────────────────────────────────
    void get_dims() {
        getmaxyx(stdscr, rows_, cols_);
    }

    void create_windows() {
        // Layout (rows_=terminal height, cols_=terminal width):
        // Row 0:        status bar (1 line)
        // Rows 1..top_h: top split — topology (left 40%) | stream (right 60%)
        // Rows top_h+1..attn_bot: attention matrix (full width)
        // Below: metrics (left 50%) | anomalies (right 50%)

        int sb     = 1;                         // status bar height
        int top_h  = std::max(10, rows_ / 3);   // topology+stream height
        int bot_h  = std::max(7,  rows_ / 4);   // metrics+anomalies height
        int attn_h = rows_ - sb - top_h - bot_h; // attention gets the rest
        attn_h     = std::max(8, attn_h);

        int topo_w   = cols_ * 40 / 100;
        int stream_w = cols_ - topo_w;
        int half_c   = cols_ / 2;

        int row = sb;

        win_topo_   = newwin(top_h,     topo_w,   row, 0);
        win_stream_ = newwin(top_h,     stream_w, row, topo_w);
        row += top_h;
        win_attn_   = newwin(attn_h,    cols_,    row, 0);
        row += attn_h;
        win_metric_ = newwin(bot_h,     half_c,   row, 0);
        win_anom_   = newwin(bot_h,     cols_ - half_c, row, half_c);

        // scroll the stream pane
        scrollok(win_stream_, TRUE);

        heights_[0] = top_h;   // topology
        heights_[1] = top_h;   // stream
        heights_[2] = attn_h;  // attention
        heights_[3] = bot_h;   // metrics
        heights_[4] = bot_h;   // anomalies
    }

    void destroy_windows() {
        for (WINDOW* w : {win_topo_, win_stream_, win_attn_, win_metric_, win_anom_}) {
            if (w) { delwin(w); }
        }
        win_topo_ = win_stream_ = win_attn_ = win_metric_ = win_anom_ = nullptr;
    }

    // ── Status bar ────────────────────────────────────────────────
    void render_statusbar() {
        attron(COLOR_PAIR(C_HEADER) | A_BOLD);
        mvhline(0, 0, ' ', cols_);
        std::string left  = " LLM-Tracer v1.0  |  Model: llama-3-8b";
        std::string right = "[Tab] Cycle Focus  [j/k] Nav  [H/L] Head  [+/-] Contrast  [q] Quit ";
        mvprintw(0, 0, "%s", left.c_str());
        mvprintw(0, cols_ - (int)right.size(), "%s", right.c_str());
        attroff(COLOR_PAIR(C_HEADER) | A_BOLD);
        wnoutrefresh(stdscr);
    }

    // ── Panel border helper ───────────────────────────────────────
    void draw_border(WINDOW* w, const std::string& title, Panel p, int panel_n) {
        bool focused = (focus_ == p);
        int  pair    = focused ? C_BORDER_F : C_DIM;

        wattron(w, COLOR_PAIR(pair) | (focused ? A_BOLD : 0));
        box(w, 0, 0);
        // Panel number + title
        std::string full = " " + std::to_string(panel_n) + ". " + title + " ";
        if (focused) full = " \u2588 " + std::to_string(panel_n) + ". " + title + " (Focus Active) ";
        mvwprintw(w, 0, 2, "%s", full.c_str());
        wattroff(w, COLOR_PAIR(pair) | A_BOLD);
    }

    // ── 1. Model Topology ─────────────────────────────────────────
    void render_topology() {
        WINDOW* w = win_topo_;
        werase(w);
        draw_border(w, "MODEL TOPOLOGY", Panel::Topology, 1);

        int max_y, max_x;
        getmaxyx(w, max_y, max_x);

        // Flatten topology into displayable rows
        flat_topo_.clear();
        flatten_node(state_.topo_root, 0);

        // Clamp selection
        topo_sel_ = std::clamp(topo_sel_, 0, (int)flat_topo_.size() - 1);

        // Scrolling
        int visible = max_y - 3;
        if (topo_sel_ >= topo_scroll_ + visible) topo_scroll_ = topo_sel_ - visible + 1;
        if (topo_sel_ <  topo_scroll_)            topo_scroll_ = topo_sel_;

        int row = 1;
        for (int i = topo_scroll_; i < (int)flat_topo_.size() && row < max_y - 1; ++i, ++row) {
            auto& [node, is_leaf] = flat_topo_[i];
            bool selected = (i == topo_sel_);
            int indent = node->depth * 2;

            if (selected && focus_ == Panel::Topology)
                wattron(w, COLOR_PAIR(C_SELECTED) | A_BOLD);
            else if (node->type == LayerType::SelfAttention)
                wattron(w, COLOR_PAIR(C_CUDA));
            else if (node->type == LayerType::MLP)
                wattron(w, COLOR_PAIR(C_INFO));
            else
                wattron(w, COLOR_PAIR(C_NORMAL));

            // Expand/collapse indicator
            std::string prefix(indent, ' ');
            std::string marker;
            if (!node->children.empty())
                marker = node->expanded ? "\u25BC " : "\u25BA ";
            else
                marker = "\u25CF ";

            // Layer type badge
            std::string badge = "";
            if (node->type != LayerType::Unknown && !node->children.empty() == false)
                badge = " [" + layer_type_str(node->type) + "]";

            std::string line = prefix + marker + node->name + badge;
            mvwprintw(w, row, 1, "%-*s", max_x - 2, line.c_str());

            wattroff(w, A_BOLD | COLOR_PAIR(C_SELECTED) | COLOR_PAIR(C_CUDA)
                                | COLOR_PAIR(C_INFO)     | COLOR_PAIR(C_NORMAL));
        }

        // Footer hint
        wattron(w, COLOR_PAIR(C_DIM));
        mvwprintw(w, max_y - 1, 2, " [j/k] Navigate  [Space] Expand/Collapse ");
        wattroff(w, COLOR_PAIR(C_DIM));

        wnoutrefresh(w);
    }

    void flatten_node(const TopoNode& node, int depth) {
        flat_topo_.push_back({const_cast<TopoNode*>(&node), node.children.empty()});
        if (node.expanded) {
            for (auto& c : node.children)
                flatten_node(c, depth + 1);
        }
    }

    // Helper: find mutable node pointer by flat index
    TopoNode* find_node(TopoNode& root, int target_idx) {
        int idx = 0;
        return find_node_r(root, idx, target_idx);
    }
    TopoNode* find_node_r(TopoNode& n, int& idx, int target) {
        if (idx++ == target) return &n;
        if (n.expanded) {
            for (auto& c : n.children) {
                auto* res = find_node_r(c, idx, target);
                if (res) return res;
            }
        }
        return nullptr;
    }

    // ── 2. Live Packet Stream ─────────────────────────────────────
    void render_stream(const std::vector<LayerPacket>& pkts) {
        WINDOW* w = win_stream_;
        werase(w);
        draw_border(w, "LIVE PACKET STREAM", Panel::Stream, 2);

        int max_y, max_x;
        getmaxyx(w, max_y, max_x);

        // Header
        wattron(w, COLOR_PAIR(C_HEADER) | A_BOLD);
        mvwprintw(w, 1, 1, " %-6s %-14s %-16s %-18s %s",
                  "ID", "TIMESTAMP", "LAYER TYPE", "DEVICE", "LATENCY");
        wattroff(w, COLOR_PAIR(C_HEADER) | A_BOLD);
        wattron(w, COLOR_PAIR(C_DIM));
        mvwhline(w, 2, 1, ACS_HLINE, max_x - 2);
        wattroff(w, COLOR_PAIR(C_DIM));

        int visible = max_y - 4;
        int start   = (int)pkts.size() > visible ? (int)pkts.size() - visible : 0;
        // stream_scroll_ offsets from end
        int show_start = std::max(0, start - stream_scroll_);
        int row = 3;

        for (int i = show_start; i < (int)pkts.size() && row < max_y - 1; ++i, ++row) {
            const auto& p = pkts[i];
            bool is_latest = (i == (int)pkts.size() - 1);
            bool is_anom   = p.is_anomaly;

            if      (is_anom)   wattron(w, COLOR_PAIR(C_WARN) | A_BOLD);
            else if (is_latest) wattron(w, COLOR_PAIR(C_FOCUSED) | A_BOLD);
            else                wattron(w, COLOR_PAIR(C_NORMAL));

            // Device color inline
            std::string dev = device_str(p.device);
            char lat_buf[16];
            std::snprintf(lat_buf, sizeof(lat_buf), "%.2fms", p.latency_ms);

            mvwprintw(w, row, 1, " %-6llu %-14s %-16s %-18s %s",
                      (unsigned long long)p.packet_id,
                      p.timestamp.c_str(),
                      layer_type_str(p.layer_type).c_str(),
                      dev.c_str(),
                      lat_buf);

            if (is_anom || is_latest) wattroff(w, A_BOLD);
            wattroff(w, COLOR_PAIR(C_WARN) | COLOR_PAIR(C_FOCUSED) | COLOR_PAIR(C_NORMAL));
        }

        wnoutrefresh(w);
    }

    // ── 3. Attention Matrix ───────────────────────────────────────
    void render_attention() {
        WINDOW* w = win_attn_;
        werase(w);

        // Find a title with head info
        std::string title = "ATTENTION MATRIX VISUALIZER";
        if (sel_pkt_ && sel_pkt_->has_attn) {
            title += " (Head " + std::to_string(sel_pkt_->attn.active_head) + "/" +
                     std::to_string(sel_pkt_->attn.num_heads - 1) + ")";
        }
        draw_border(w, title, Panel::Attention, 3);

        int max_y, max_x;
        getmaxyx(w, max_y, max_x);

        if (!sel_pkt_ || !sel_pkt_->has_attn) {
            wattron(w, COLOR_PAIR(C_DIM));
            mvwprintw(w, 2, 4, "No attention data — select an attention layer in panel 1.");
            wattroff(w, COLOR_PAIR(C_DIM));
            wnoutrefresh(w);
            return;
        }

        const auto& attn  = sel_pkt_->attn;
        int  head         = attn.active_head;
        int  n            = attn.num_tokens;
        int  cell_w       = 5; // characters per cell

        // Clamp viewport
        int visible_cols = (max_x - 10) / cell_w;
        int visible_rows = max_y - 5;
        attn_vp_x_ = std::clamp(attn_vp_x_, 0, std::max(0, n - visible_cols));
        attn_vp_y_ = std::clamp(attn_vp_y_, 0, std::max(0, n - visible_rows));

        // Token header row
        wattron(w, COLOR_PAIR(C_INFO) | A_BOLD);
        mvwprintw(w, 1, 10, "Tokens: ");
        int col_x = 10;
        for (int c = attn_vp_x_; c < std::min(n, attn_vp_x_ + visible_cols); ++c) {
            std::string tok = attn.token_labels[c];
            if ((int)tok.size() > cell_w - 1) tok = tok.substr(0, cell_w - 1);
            mvwprintw(w, 1, col_x, "[%-*s]", cell_w - 2, tok.c_str());
            col_x += cell_w;
        }
        wattroff(w, COLOR_PAIR(C_INFO) | A_BOLD);

        // Matrix rows
        for (int r = attn_vp_y_; r < std::min(n, attn_vp_y_ + visible_rows); ++r) {
            int ry = 2 + (r - attn_vp_y_);
            if (ry >= max_y - 1) break;

            // Row label
            wattron(w, COLOR_PAIR(C_INFO));
            std::string lbl = "[" + attn.token_labels[r] + "]";
            mvwprintw(w, ry, 1, "%-8s", lbl.c_str());
            wattroff(w, COLOR_PAIR(C_INFO));

            col_x = 10;
            for (int c = attn_vp_x_; c < std::min(n, attn_vp_x_ + visible_cols); ++c) {
                float v    = attn.weights[head][r][c];
                float adj  = std::pow(v * attn_contrast_, 0.6f);
                const char* block = weight_char(adj);
                int pair  = weight_color(adj);
                wattron(w, COLOR_PAIR(pair));
                mvwprintw(w, ry, col_x, "%-*s", cell_w, block);
                wattroff(w, COLOR_PAIR(pair));
                col_x += cell_w;
            }
        }

        // Viewport legend (bottom right)
        wattron(w, COLOR_PAIR(C_DIM));
        std::string vp = "Viewport [" + std::to_string(attn_vp_x_) + "-" +
                         std::to_string(attn_vp_x_ + visible_cols - 1) + "] x [" +
                         std::to_string(attn_vp_y_) + "-" +
                         std::to_string(attn_vp_y_ + visible_rows - 1) + "]";
        mvwprintw(w, max_y - 2, max_x - (int)vp.size() - 2, "%s", vp.c_str());

        std::string hint = "[h/j/k/l] Pan  [H/L] Head  [+/-] Contrast ";
        mvwprintw(w, max_y - 1, max_x - (int)hint.size() - 2, "%s", hint.c_str());
        wattroff(w, COLOR_PAIR(C_DIM));

        wnoutrefresh(w);
    }

    // Attention weight → block character
    const char* weight_char(float v) {
        if (v > 0.85f)      return "\u2588\u2588\u2588\u2588"; // ████
        else if (v > 0.65f) return "\u2593\u2593\u2593\u2593"; // ▓▓▓▓
        else if (v > 0.40f) return "\u2592\u2592\u2592\u2592"; // ▒▒▒▒
        else if (v > 0.20f) return "\u2591\u2591\u2591\u2591"; // ░░░░
        else                return "    ";
    }

    int weight_color(float v) {
        if (v > 0.70f) return C_ATTN_HI;
        if (v > 0.35f) return C_ATTN_MED;
        return C_NORMAL;
    }

    // ── 4. Runtime Metrics ────────────────────────────────────────
    void render_metrics() {
        WINDOW* w = win_metric_;
        werase(w);
        draw_border(w, "RUNTIME METRICS INSPECTOR", Panel::Metrics, 4);

        int max_y_unused, max_x;
        getmaxyx(w, max_y_unused, max_x);
        (void)max_y_unused;

        if (!sel_pkt_) {
            mvwprintw(w, 2, 3, "Waiting for data...");
            wnoutrefresh(w);
            return;
        }

        const auto& p = *sel_pkt_;
        int row = 1;

        // Layer name
        wattron(w, COLOR_PAIR(C_INFO) | A_BOLD);
        mvwprintw(w, row++, 2, "Layer : %s", p.layer_name.c_str());
        wattroff(w, COLOR_PAIR(C_INFO) | A_BOLD);

        wattron(w, COLOR_PAIR(C_NORMAL));
        mvwprintw(w, row++, 2, "Type  : %s", layer_type_str(p.layer_type).c_str());
        mvwprintw(w, row++, 2, "Device: %s", device_str(p.device).c_str());

        row++;
        wattron(w, COLOR_PAIR(C_FOCUSED) | A_BOLD);
        mvwprintw(w, row++, 2, "Tensor Shape : %s   Dtype: %s",
                  p.output_shape.to_string().c_str(),
                  p.output_shape.dtype.c_str());
        wattroff(w, COLOR_PAIR(C_FOCUSED) | A_BOLD);

        // Sparsity bar
        int bar_len = max_x - 20;
        std::string bar = sparsity_bar(p.sparsity_pct, bar_len);
        mvwprintw(w, row, 2, "Sparsity      : ");
        wattron(w, COLOR_PAIR(p.sparsity_pct > 70 ? C_WARN : C_GREEN) | A_BOLD);
        mvwprintw(w, row, 18, "%s %.1f%%", bar.c_str(), p.sparsity_pct);
        wattroff(w, COLOR_PAIR(C_WARN) | COLOR_PAIR(C_GREEN) | A_BOLD);
        row++;

        // Activation stats
        mvwprintw(w, row++, 2, "Act. Mean/Max/Min : %.4f / %.4f / %.4f",
                  p.activation_mean, p.activation_max, p.activation_min);

        // Latency
        const char* bound = (p.latency_ms > 120.0) ? "  \u26A0 High" : "  \u2713 Normal";
        int lat_color = (p.latency_ms > 120.0) ? C_WARN : C_GREEN;
        mvwprintw(w, row, 2, "Latency Delta : %.3f ms", p.latency_ms);
        wattron(w, COLOR_PAIR(lat_color) | A_BOLD);
        mvwprintw(w, row, 30, "%s", bound);
        wattroff(w, COLOR_PAIR(lat_color) | A_BOLD);
        row++;

        wattroff(w, COLOR_PAIR(C_NORMAL));
        wnoutrefresh(w);
    }

    std::string sparsity_bar(double pct, int len) {
        int filled = (int)(pct / 100.0 * len);
        std::string bar;
        for (int i = 0; i < len; ++i) {
            if (i < filled) bar += "\U0001F7E9"; // 🟩
            else             bar += "\u2B1C";     // ⬜
        }
        return bar;
    }

    // ── 5. Anomaly Ledger ─────────────────────────────────────────
    void render_anomalies(const std::vector<AnomalyEntry>& anoms) {
        WINDOW* w = win_anom_;
        werase(w);
        draw_border(w, "NUMERICAL ANOMALY LEDGER", Panel::Anomalies, 5);

        int max_y, max_x;
        getmaxyx(w, max_y, max_x);

        int visible = max_y - 2;
        int start   = (int)anoms.size() > visible ? (int)anoms.size() - visible : 0;
        int row     = 1;

        if (anoms.empty()) {
            wattron(w, COLOR_PAIR(C_GREEN) | A_BOLD);
            mvwprintw(w, row, 2, "\u2713 No anomalies detected.");
            wattroff(w, COLOR_PAIR(C_GREEN) | A_BOLD);
        }

        for (int i = start; i < (int)anoms.size() && row < max_y - 1; ++i, ++row) {
            const auto& a = anoms[i];
            const char* icon;
            int  pair;
            if (a.severity == AnomalySeverity::ERROR) {
                icon = "\u2716"; pair = C_ERROR;
            } else if (a.severity == AnomalySeverity::WARN) {
                icon = "\u26A0"; pair = C_WARN;
            } else {
                icon = "\u2139"; pair = C_INFO;
            }
            wattron(w, COLOR_PAIR(pair));
            mvwprintw(w, row, 1, " %s %s %s",
                      a.timestamp.c_str(), icon, a.message.c_str());
            // Truncate to window width
            wattroff(w, COLOR_PAIR(pair));
        }
        wnoutrefresh(w);
    }

    // ── Input handlers ────────────────────────────────────────────
    void cycle_focus() {
        int f = (int)focus_;
        focus_ = (Panel)((f + 1) % 5);
    }

    void on_down() {
        switch (focus_) {
            case Panel::Topology:
                if (topo_sel_ < (int)flat_topo_.size() - 1) ++topo_sel_;
                break;
            case Panel::Stream:
                if (stream_scroll_ > 0) --stream_scroll_;
                break;
            case Panel::Attention:
                ++attn_vp_y_;
                break;
            default: break;
        }
    }

    void on_up() {
        switch (focus_) {
            case Panel::Topology:
                if (topo_sel_ > 0) --topo_sel_;
                break;
            case Panel::Stream:
                ++stream_scroll_;
                break;
            case Panel::Attention:
                if (attn_vp_y_ > 0) --attn_vp_y_;
                break;
            default: break;
        }
    }

    void on_left() {
        if (focus_ == Panel::Attention && attn_vp_x_ > 0) --attn_vp_x_;
    }

    void on_right() {
        if (focus_ == Panel::Attention) ++attn_vp_x_;
    }

    void on_space() {
        if (focus_ == Panel::Topology && !flat_topo_.empty()) {
            TopoNode* node = find_node(state_.topo_root, topo_sel_);
            if (node && !node->children.empty())
                node->expanded = !node->expanded;
        }
    }

    // ── Members ───────────────────────────────────────────────────
    SimState& state_;
    Panel     focus_;

    WINDOW* win_topo_   = nullptr;
    WINDOW* win_stream_ = nullptr;
    WINDOW* win_attn_   = nullptr;
    WINDOW* win_metric_ = nullptr;
    WINDOW* win_anom_   = nullptr;

    int rows_ = 24, cols_ = 80;
    int heights_[5] = {};

    // Topology state
    std::vector<std::pair<TopoNode*, bool>> flat_topo_;
    int topo_sel_    = 0;
    int topo_scroll_ = 0;

    // Stream state
    int stream_scroll_ = 0;

    // Attention state
    int   attn_vp_x_     = 0;
    int   attn_vp_y_     = 0;
    float attn_contrast_ = 1.0f;

    // Selected packet for metrics/attention
    LayerPacket  latest_pkt_;
    LayerPacket* sel_pkt_       = nullptr;
    bool         sel_pkt_locked_ = false;  // future: lock to a specific packet
};
