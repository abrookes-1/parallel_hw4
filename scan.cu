#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

/*

WRITE CUDA KERNEL FOR COUNT HERE

*/
const int reduc_thread_elements = 2;


__global__ void parallel_reduc(int *data,int *output, int *values, int *iteration){
    int start = blockDim.x * blockIdx.x + threadIdx.x;
    int accumulator = 0;
    for (int i=0; i<reduc_thread_elements; i++){
        accumulator += data[start + i];
    }
    output[start+reduc_thread_elements-1] = accumulator;
//    output[];
}

__global__ void parallel_reduc_2(int *data){
    int start = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    for (int i=blockDim.x/2; i>0; i>>=1){
        if (tid < i){
            data[start] += data[start+i];
        }
        __syncthreads();
    }
    if (tid == 0){
        data[blockIdx.x] = data[start];
    }
}

__global__ void parallel_reduc_3(int *data, int *out, int *values){
    extern __shared__ float chunk_data[1024];

    int start = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int indx = start*2+1;
    int gap = 1;
    int start_size = *values >> 1;

    chunk_data[2*tid] = data[2*tid];
    chunk_data[2*tid+1] = data[2*tid+1];
//    out[indx] = data[indx] + data[indx-gap];
    for (int span=start_size; span>0; span>>=1){
        __syncthreads();
        if (tid < span){
            int from = gap*(2*tid+1)-1;
            int to = gap*(2*tid+2)-1;

            chunk_data[to] += chunk_data[from];
        }
        gap = gap * 2;
    }
    __syncthreads();
    out[2*tid] = chunk_data[2*tid];
    out[2*tid+1] = chunk_data[2*tid+1];
//    out[2*tid] = 1;
//    out[2*tid+1] = 2;
}

__global__ void up_sweep(int *data,int *output, int *values){

}

// write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int * serial_implementation(int * data, int vals) {
    int * output = (int *)malloc(sizeof(int) * vals);
    
    output[0] = 0;
    for (int i = 1; i < vals; i++) {
        output[i] = output[i-1] + data[i-1];
    }
    
    return output;
}

int main(int argc, char ** argv) {
    
    assert(argc == 2);
    int values = atoi(argv[1]); // Values is guaranteed to be no more than 10000000
    int zeros;
    if (values % 1024 == 0){
        zeros = 0;
    } else{
        zeros = 1024 - (values % 1024);
    }

    int * data = (int *)malloc(sizeof(int) * (values+zeros));

    // Generate "random" vector
    std::mt19937 gen(13); // Keep constant to maintain determinism between runs
    std::uniform_int_distribution<> dist(0, 50);
    for (int i = 0; i < values; i++) {
        data[i] = dist(gen);
    }
    for (int i=1; i<=zeros; i++){
        data[values + i] = 0;
    }
    int z_values = values + zeros;

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int * h_output = (int *)malloc(sizeof(int) * values); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

    // PERFORM NECESSARY VARIABLE DECLARATIONS HERE
    int *data_p, *values_p;
    cudaMallocManaged(&values_p, sizeof(int));
    cudaMallocManaged(&data_p, z_values * sizeof(int));
    cudaMallocManaged(&h_output, z_values * sizeof(int));

    // PERFORM NECESSARY DATA TRANSFER HERE
    *values_p = z_values;
    for (int i=0; i<z_values; i++){
        data_p[i] = data[i];
//        h_output[i] = data[i];
        h_output[i] = 0;
    }

    cudaEventRecord(begin, stream);

    // LAUNCH KERNEL HERE
    int threads_per_block = 512;
    int elements_per_block = threads_per_block * reduc_thread_elements;
    // ceiling of values / elements_per_block
    int num_blocks = (values + elements_per_block - 1) / elements_per_block;
    dim3 grid_dim_reduc(num_blocks, 1, 1);
    dim3 block_dim_reduc(threads_per_block, 1, 1);
//    parallel_reduc <<<grid_dim_reduc, block_dim_reduc>>> (data_p, h_output, values_p, );
//    up_sweep <<<grid_dim, block_dim>>> (data_p, h_output, values_p);
//    reduce <<<32, 32>>> (data_p, h_output);
//    parallel_reduc_3 <<<z_values/1024, 512>>> (data_p, h_output, values_p);
    parallel_reduc_3 <<<1, 512>>> (data_p, h_output, values_p);
    cudaEventRecord(end, stream);

    /* 

    PERFORM NECESSARY DATA TRANSFER HERE

    */

    cudaStreamSynchronize(stream);
    for (int i=0; i<15; i++){
        printf("data: %i\n", data[i]);
    }
    for (int i=0; i<15; i++){
        printf("out: %i\n", h_output[i]);
    }
    printf("first: %i\n", h_output[0]);
    printf("last: %i\n", h_output[z_values-1]);
    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    //DEALLOCATE RESOURCES HERE
    cudaFree(data_p);
    cudaFree(values_p);
//    cudaFree(h_output);

//    for (int i=0; i<values; i++){
//        printf("val: %i\n", h_output[i]);
//    }

    int * reference_output = serial_implementation(data, values);
    for (int i = 0; i < values; i++) {
        if (reference_output[i] != h_output[i]) {
            printf("ERROR: %d != %d at index %d\n", reference_output[i], h_output[i], i);
            abort();
        }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(data);
    free(reference_output);
    free(h_output);
    cudaFree(h_output);

    return 0;
}
