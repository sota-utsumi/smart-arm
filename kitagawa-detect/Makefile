CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
TARGET := build/weed_detector_cpp
SRC := weed_detector.cpp
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null)

.PHONY: all setup clean check-opencv

all: $(TARGET)

setup:
	sudo apt-get update
	sudo apt-get install -y build-essential pkg-config libopencv-dev

check-opencv:
	@pkg-config --exists opencv4 || (echo "OpenCV development files not found. Run 'make setup' on Ubuntu." && exit 1)

build:
	@mkdir -p build

$(TARGET): $(SRC) | build check-opencv
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) $(SRC) -o $(TARGET) $(OPENCV_LIBS)

clean:
	rm -rf build
