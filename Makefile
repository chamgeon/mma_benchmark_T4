SRCS = src/main.cu src/kernel_1.cu src/kernel_2.cu src/kernel_3.cu src/kernel_4.cu src/kernel_5.cu src/kerenl_6.cu src/kernel_7.cu src/cublas_benchmark.cu
TARGET = bin/runner

$(TARGET): $(SRCS)
	nvcc -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)