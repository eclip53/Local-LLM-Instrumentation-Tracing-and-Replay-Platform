CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude
LDFLAGS  := -lncursesw -lpthread

TARGET   := llm-tracer
SRC      := src/main.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) include/types.h include/ring_buffer.h include/simulator.h include/tui.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
