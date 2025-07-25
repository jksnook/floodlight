CXX := g++
TARGET := nnue-chess
TMPDIR := .tmp

CXXFLAGS := -std=c++23 -O3 -march=native
NAME := floodlight

SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(patsubst %.cpp,$(TMPDIR)/%.o,$(SOURCES))
DEPENDS := $(patsubst %.cpp,$(TMPDIR)/%.d,$(SOURCES))

all: $(TARGET)
clean:
	rm -rf $(TMPDIR) *.o
debug: CXXFLAGS += -g -Wall
debug: all

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $(NAME)

$(TMPDIR)/%.o: %.cpp Makefile | $(TMPDIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

-include $(DEPENDS)

$(TMPDIR):
	mkdir "$(TMPDIR)" "$(TMPDIR)/src"