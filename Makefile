NVCC = nvcc
FLAGS = -I ~/cutlass/include -arch=sm_75 -O3 -Xptxas -dlcm=cg
SRCS = src/main.cu src/kernel_1.cu src/kernel_2.cu src/kernel_3.cu src/kernel_4.cu src/kernel_5.cu src/kernel_6.cu src/kernel_7.cu src/cublas_benchmark.cu
TARGET = runner_cg

$(TARGET): $(SRCS)
	$(NVCC) $(FLAGS) $(SRCS) -o $(TARGET) -lcublas

clean:
	rm -f $(TARGET)