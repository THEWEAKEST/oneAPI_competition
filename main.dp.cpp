#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctime>
#include <dpct/blas_utils.hpp>

// CUDA runtime
// #include <unistd.h>
// #include "cudnn/cudnn.h"

#include "dense_help_func.dp.cpp"
#include "conv.dp.cpp"
#include <chrono>

// #define DEBUG
#define size_matrix int
#define element_type float
#define CUDNN
// #define CPU_SERIAL
#define GPU_PARALLEL

void generate_matrix(element_type *mat, int m, int n);
void generate_filter(element_type *mat, int size);

int main()
{
    clock_t clk ;
    clk = clock();
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    // 初始化CPU数据 32 16 30 14
    int N;
    printf("Input the batch size:");
    scanf("%d",&N);
    const int inC = 6; // inChannel >15会出错？
    int inH = 768;
    int inW = 512;
    printf("\nInput the H and W:");
    scanf("%d %d",&inH,&inW);
    printf("%d %d\n",inH,inW);
    const int kernelH = 6;
    const int kernelW = 6;
    const int outC = 6; // outChannel 每个都与不同的卷积核运算 之后再分别放到outChannel中
    const int outH = inH - kernelH + 1;
    const int outW = inW - kernelW + 1;

    // cudaSetDevice(7);

    element_type *inputs, *outputs, *kernel;
    int in_size = N * inC * inH * inW,
        out_size = N * outC * outH * outW,
        filter_size = outC * inC * kernelH * kernelW;
    inputs = (element_type *)malloc(in_size * sizeof(element_type));
    outputs = (element_type *)malloc(out_size * sizeof(element_type));
    kernel = (element_type *)malloc(filter_size * sizeof(element_type));
    for (int i = 0; i < in_size; i++)
    {
        inputs[i] = rand() % 10;
    }
    for (int i = 0; i < filter_size; i += 3)
    {
        kernel[i] = -1;
        kernel[i + 1] = 0;
        kernel[i + 2] = 1;
        // kernel[i + 3] = 1;
    }
    for (int i = 0; i < out_size; i++)
    {
        outputs[i] = 0;
    }
    // 计时数据
    dpct::event_ptr start, stop;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    start = new sycl::event();
    stop = new sycl::event();
    int iters = 100;
    float msecTotal = 0;
    double msecPerMatrixMul[2] = {0, 0}, gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = out_size * inC * kernelH * kernelW;




#ifdef CPU_SERIAL
    /* ---- CPU serial BEGIN ---- */
    float *self_outputs;
    self_outputs = (element_type *)malloc(out_size * sizeof(element_type));
    for (int i = 0; i < out_size; i++)
    {
        self_outputs[i] = 0;
    }
    serial_convolution(inputs, self_outputs, kernel, N, inC, inH, inW, outC, outH, outW, kernelH, kernelW);
    /*
     for (int i = 0; i < outH * outW; i++)
     {
         //printf("%.2f|%.2f\n", outputs[i], self_outputs[i]);
         printf("%.2f ",self_outputs[i]);
         if(( i + 1 ) % outW == 0)
             printf("\n");
     }
    */
   // msecTotal = std::chrono::duration<float, std::milli>( stop_ct1 - start_ct1 ).count();
   // printf("conv cost time: %f\n", msecTotal / (iters - 1));
    printf("%lf\n" ,(double)(clock() - clk)/(double)CLOCKS_PER_SEC);
    /* ---- CPU serial END ---- */
#endif

#ifdef GPU_PARALLEL
    /* ---- SELF CUDA BEGIN ---- */
    auto alpha = 1.0f;
    auto beta = 0.0f;
    size_t input_size = N * inC * inH * inW * sizeof(float);
    size_t kernel_size = outC * inC * kernelH * kernelW * sizeof(float);
    size_t output_size = N * outC * outH * outW * sizeof(float);
    
    // 初始化GPU数据
    
    printf("BLOCK 1\n");
    
    float *self_outputs;
    self_outputs = (element_type *)malloc(out_size * sizeof(element_type));
    element_type *self_dev_input, *self_dev_kernel, *self_dev_output;
    self_dev_input = (float *)sycl::malloc_device(input_size, q_ct1);
    self_dev_kernel = (float *)sycl::malloc_device(kernel_size, q_ct1);
    self_dev_output = (float *)sycl::malloc_device(output_size, q_ct1);
    q_ct1.memcpy(self_dev_input, inputs, input_size).wait();
    q_ct1.memcpy(self_dev_kernel, kernel, kernel_size).wait();
    q_ct1.memcpy(self_dev_output, self_outputs, output_size).wait();
    
    printf("BLOCK 2\n");

    const int THREAD_HEIGHT = 1, THREAD_WIDTH = 1,                                         // 一个线程负责的元素数
        KERNEL_HEIGHT = kernelH, KERNEL_WIDTH = kernelW,                                   // 卷积核大小
        BLOCK_HEIGHT = 8, BLOCK_WIDTH = 4,                                                 // 分块大小
        MALLOC_KERNEL_HEIGHT = KERNEL_HEIGHT % 2 == 0 ? KERNEL_HEIGHT : KERNEL_HEIGHT + 1, // 用于kernel在SMEM的修正尺寸 奇数尺寸无法分配空间
        MALLOC_KERNEL_WIDTH = KERNEL_WIDTH % 2 == 0 ? KERNEL_WIDTH : KERNEL_WIDTH + 1,     // 用于kernel在SMEM的修正尺寸
        MALLOC_BLOCK_HEIGHT = (BLOCK_HEIGHT + KERNEL_HEIGHT) * 2,                          // 用于block在SMEM的修正尺寸
        MALLOC_BLOCK_WIDTH = (BLOCK_WIDTH + KERNEL_WIDTH) * 2,                             // 用于block在SMEM的修正尺寸
        MALLOC_TEMP_SIZE = outC * 4;                                                       // 用于计算时暂存计算结果的寄存器大小

    // printf("%d %d %d %d %d\n",KERNEL_HEIGHT,KERNEL_WIDTH,MALLOC_BLOCK_HEIGHT,MALLOC_BLOCK_WIDTH,MALLOC_TEMP_SIZE);

    // 第一个参数是x轴范围，第二个是y轴
    
    printf("BLOCK 3\n");
    
    sycl::range<3> dimGrid(1, outH / BLOCK_HEIGHT, outW / BLOCK_WIDTH);
    sycl::range<3> dimBlock(1, BLOCK_HEIGHT / THREAD_HEIGHT,
                            BLOCK_WIDTH / THREAD_WIDTH);

    // cudaEventRecord(start);
    
    printf("BLOCK 4\n");

    for (int run = 0; run < iters; run++)
    {
        /*
        DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 2> s_kernel_acc_ct1(
                sycl::range<2>(MALLOC_KERNEL_HEIGHT, MALLOC_KERNEL_WIDTH), cgh);
            sycl::local_accessor<float, 2> s_in_acc_ct1(
                sycl::range<2>(MALLOC_BLOCK_HEIGHT, MALLOC_BLOCK_WIDTH), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<3> item_ct1) {
                    v1_convolution<BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_HEIGHT,
                                   KERNEL_WIDTH, MALLOC_TEMP_SIZE,
                                   MALLOC_KERNEL_HEIGHT, MALLOC_KERNEL_WIDTH,
                                   MALLOC_BLOCK_HEIGHT, MALLOC_BLOCK_WIDTH>(
                        self_dev_input, self_dev_output, self_dev_kernel, N,
                        inC, inH, inW, outC, outH, outW, kernelH, kernelW,
                        item_ct1, s_kernel_acc_ct1, s_in_acc_ct1);
                });
        });
        if (run == 1)
            /*
            DPCT1012:48: Detected kernel execution time measurement pattern and
            generated an initial code for time measurements in SYCL. You can
            change the way time is measured depending on your goals.
            */
            start_ct1 = std::chrono::steady_clock::now();
    }

    /*
    DPCT1012:43: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    
    printf("BLOCK 5\n");
    
    stop_ct1 = std::chrono::steady_clock::now();

    q_ct1.memcpy(self_outputs, self_dev_output, output_size).wait();

    msecTotal =std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    printf("my conv cost time: %f\n", msecTotal / (iters - 1));
    msecPerMatrixMul[1] = msecTotal / (iters - 1);
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    /*
    for (int i = 0; i < outC * outH * outW; i++)
    {
        if (outputs[i] != -self_outputs[i])
        {
            printf("WRONG VALUE: %.2f|%.2f at %d\n", outputs[i], -self_outputs[i], i);
            break;
        }
    }
    */
    /* ---- SELF CUDA END ---- */
#endif

    // _exit(0);
    //sycl::free(dev_input, q_ct1);
    //sycl::free(dev_kernel, q_ct1);
    //sycl::free(dev_output, q_ct1);
    //sycl::free(self_dev_input, q_ct1);
    //sycl::free(self_dev_kernel, q_ct1);
    //sycl::free(self_dev_output, q_ct1);
    free(self_outputs);
    free(inputs);
    free(outputs);
    free(kernel);
    return 0;
}

void generate_matrix(element_type *mat, int m, int n)
{

    for (int i = 0; i < m * n; i++)
    {
        // printf("total %d\n", m * n);
        mat[i] = rand() % 10;
        // printf("1\n");
        // int row = (i / n), col = (i % n);                         // 行数与列数
        // int row_block = row / block_m, col_block = col / block_n; // 块行号与列号
        // if ((row_block * k_block + col_block) % stride == 0)
        // {
        //     mat[i] = 1;
        // }
        // else
        // {
        //     mat[i] = 0;
        // }
    }
}

void generate_filter(element_type *mat, int size)
{
    if (size == 3)
    {
        for (int i = 0; i <= 6; i += 3)
        {
            mat[i] = 1;
            mat[i + 1] = 0;
            mat[i + 2] = -1;
        }
    }
}
