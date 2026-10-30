NVCC = nvcc
FLAGS = -I ~/cutlass/include -arch=sm_75 -O3
SRCS = src/main.cu src/kernel_1.cu src/kernel_2.cu src/kernel_3.cu src/kernel_4.cu src/kernel_5.cu src/kernel_6.cu src/kernel_7.cu src/kernel_8.cu src/cublas_benchmark.cu
KERNEL_SRCS = $(filter src/kernel_%.cu, $(SRCS))
BUILD = build
OBJS = $(patsubst src/%.cu, $(BUILD)/%.o, $(SRCS))
TARGET = runner

# Each .cu compiles to its own object, so only changed files rebuild and `make -j` runs them in parallel.
$(TARGET): $(OBJS)
	$(NVCC) -arch=sm_75 $(OBJS) -o $(TARGET) -lcublas

$(BUILD)/%.o: src/%.cu | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@

# Kernel runners include sizes.h, so changing the size list rebuilds them.
$(patsubst src/%.cu, $(BUILD)/%.o, $(KERNEL_SRCS)): src/sizes.h

$(BUILD):
	mkdir -p $(BUILD)

.PHONY: clean
clean:
	rm -rf $(TARGET) $(BUILD)
