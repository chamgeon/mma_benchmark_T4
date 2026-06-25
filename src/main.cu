#include <cstdlib>
#include <cstdio>

void run_cublas(int argc, char **argv);
void run_kernel_1(int argc, char **argv);
void run_kernel_2(int argc, char **argv);
void run_kernel_3(int argc, char **argv);
void run_kernel_4(int argc, char **argv);
void run_kernel_5(int argc, char **argv);
void run_kernel_6(int argc, char **argv);
void run_kernel_7(int argc, char **argv);

int main(int argc, char **argv){
    if (argc != 2){
        printf("Usage: %s <kernel_number>\n", argv[0]);
        printf("  Kernels: 1,2, ..., 7, 99(cublas)\n");
        return 1;
    }

    int kernel_num = atoi(argv[1]);

    switch(kernel_num){
        case 1: run_kernel_1(argc - 1, argv + 1); break;
        case 2: run_kernel_2(argc - 1, argv + 1); break;
        case 3: run_kernel_3(argc - 1, argv + 1); break;
        case 4: run_kernel_4(argc - 1, argv + 1); break;
        case 5: run_kernel_5(argc - 1, argv + 1); break;
        case 6: run_kernel_6(argc - 1, argv + 1); break;
        case 7: run_kernel_7(argc - 1, argv + 1); break;
        case 99: run_cublas(argc - 1, argv + 1); break;
        default:
            printf("Unknown kernel: %d\n", kernel_num);
            return 1;
    }
    
    return 0;
}
