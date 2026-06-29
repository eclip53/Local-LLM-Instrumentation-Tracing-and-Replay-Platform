#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

// ─────────────────────────────────────────────────────────
//  Fixed-size ring buffer  (lock-free for single producer /
//  single consumer; good enough for our TUI use-case)
// ─────────────────────────────────────────────────────────

template<typename T, std::size_t N>
class RingBuffer {
    static_assert(N > 0, "RingBuffer size must be > 0");
public:
    RingBuffer() : head_(0), tail_(0), count_(0) {}

    // Append item; if full, oldest is overwritten.
    void push(T item) {
        buf_[head_] = std::move(item);
        head_ = (head_ + 1) % N;
        if (count_ < N) {
            ++count_;
        } else {
            // overwrite: advance tail
            tail_ = (tail_ + 1) % N;
        }
    }

    // Returns all items in insertion order (oldest first).
    std::vector<T> snapshot() const {
        std::vector<T> out;
        out.reserve(count_);
        for (std::size_t i = 0; i < count_; ++i) {
            out.push_back(buf_[(tail_ + i) % N]);
        }
        return out;
    }

    std::size_t size()  const { return count_; }
    bool        empty() const { return count_ == 0; }
    void        clear() { head_ = tail_ = count_ = 0; }

    // Access by logical index (0 = oldest).
    const T& at(std::size_t idx) const {
        if (idx >= count_) throw std::out_of_range("RingBuffer::at");
        return buf_[(tail_ + idx) % N];
    }

    // Most recently pushed item.
    const T& back() const {
        if (count_ == 0) throw std::out_of_range("RingBuffer::back");
        return buf_[(head_ + N - 1) % N];
    }

private:
    std::array<T, N> buf_;
    std::size_t head_, tail_, count_;
};
