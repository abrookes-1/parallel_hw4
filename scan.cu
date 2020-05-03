#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

/*

WRITE CUDA KERNEL FOR COUNT HERE

*/

__global__ void scan_block_3(int *data, int *out, int *values, int *block_sums){
    __shared__ int chunk_data[1024];

    int start = 1024 * blockIdx.x;
    int tid = threadIdx.x;
    int gap = 1;
    int start_size = 1024 >> 1;
    int temp_block_sum = 0;

    chunk_data[2*tid] = data[start+(2*tid)];
    chunk_data[2*tid+1] = data[start+(2*tid+1)];

    __syncthreads();
    if (threadIdx.x == 0){
        temp_block_sum = chunk_data[1023];
    }

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
//    out[start+(2*tid)] = chunk_data[2*tid];
//    out[start+(2*tid+1)] = chunk_data[2*tid+1];

    // sweep down tree
    if (tid == 0){
        chunk_data[1023] = 0;
    }

    for (int nodes=1; nodes<1024; nodes*=2){
        gap >>= 1;
        __syncthreads();
        if (tid < nodes){
            int from = gap*(2*tid+1)-1;
            int to = gap*(2*tid+2)-1;

            int temp = chunk_data[from];
            chunk_data[from] = chunk_data[to];
            chunk_data[to] += temp;
        }
    }
    __syncthreads();

    // write back to unified memory
    out[start+(2*tid)] = chunk_data[2*tid];
    out[start+(2*tid+1)] = chunk_data[2*tid+1];

    // write to block sums
    if (threadIdx.x == 0){
        block_sums[blockIdx.x] += temp_block_sum + chunk_data[1023];
    }
}

__global__ void apply_block_sum(int *data, int *out, int *values, int *block_sums){
    //__shared__ float chunk_data[1024];
//    __shared__ int val_to_add;
    int tid = threadIdx.x;
    int start = 1024 * blockIdx.x;
    int ind_a = start+(tid*2);
    int ind_b = start+(tid*2)+1;

//    if (threadIdx.x == 0) {
//        if (blockIdx.x == 0) {
//            val_to_add = 0;
//        } else {
//            val_to_add = block_sums[blockIdx.x - 1];
//        }
//    }
//    __syncthreads();
    int val_to_add = block_sums[blockIdx.x];

    out[ind_a] = data[ind_a] + val_to_add;
    out[ind_b] = data[ind_b] + val_to_add;

//    // place into shared
//    chunk_data[2*tid] = data[start+(2*tid)];
//    chunk_data[2*tid+1] = data[start+(2*tid+1)];
//    __syncthreads();
//
//    chunk_data[]
//
//    __syncthreads();
//    // write back to unified memory
//    out[start+(2*tid)] = chunk_data[2*tid];
//    out[start+(2*tid+1)] = chunk_data[2*tid+1];
}

//__global__ void down_sweep(int *data,int *output, int *values){
//    int
//}


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

    int * h_output = (int *)malloc(sizeof(int) * (values+zeros)); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END
    // PERFORM NECESSARY VARIABLE DECLARATIONS HERE
    int blocks = z_values/1024;
    if (blocks % 1024 == 0){
        zeros = 0;
    } else {
        zeros = 1024 - (blocks % 1024);
    }
    int blocks_padded = blocks + zeros;
    printf("blocks padded: %i\n", blocks_padded);
    printf("blocks: %i\n", blocks);

    int *data_p, *values_p, *intermediate, *num_blocks, *dummy, *block_sums, *block_sums_scanned, *bb_sums, *bb_sums_scanned, *bb_len, *blocks_intermediate;
//    int * block_sums = (int *)malloc(sizeof(int) * blocks_padded);
//    int * block_sums_scanned = (int *)malloc(sizeof(int) * blocks_padded);

    cudaMallocManaged(&values_p, sizeof(int));
    cudaMallocManaged(&num_blocks, sizeof(int));
    cudaMallocManaged(&data_p, z_values * sizeof(int));
    cudaMallocManaged(&intermediate, z_values * sizeof(int));
    cudaMallocManaged(&h_output, z_values * sizeof(int));
    cudaMallocManaged(&block_sums, blocks_padded * sizeof(int));
    cudaMallocManaged(&blocks_intermediate, blocks_padded * sizeof(int));
    cudaMallocManaged(&block_sums_scanned, blocks_padded * sizeof(int));
    cudaMallocManaged(&dummy, blocks_padded/1024 * sizeof(int));
    cudaMallocManaged(&bb_sums, 1024 * sizeof(int));
    cudaMallocManaged(&bb_sums_scanned, 1024 * sizeof(int));
    cudaMallocManaged(&bb_len, sizeof(int));

    // PERFORM NECESSARY DATA TRANSFER HERE
    *values_p = z_values;
    *num_blocks = z_values/1024;
    *bb_len = 1024;
    for (int i=0; i<z_values; i++){
        data_p[i] = data[i];
    }
    for (int i=0; i<blocks_padded; i++){
        block_sums[i] = 0;
        block_sums_scanned[i] = 0;
    }
    for (int i=0; i<1024; i++){
        bb_sums[i] = 0;
        bb_sums_scanned[i] = 0;
    }

    cudaEventRecord(begin, stream);

    // LAUNCH KERNEL HERE
    // scan data in blocks of 1024
    scan_block_3 <<<z_values/1024, 512>>> (data_p, intermediate, values_p, block_sums);
    cudaDeviceSynchronize();

    // scan block sums array
    printf("blocks padded: %i\n", blocks_padded);
    scan_block_3 <<<blocks_padded/1024, 512>>> (block_sums, block_sums_scanned, num_blocks, bb_sums);
    cudaDeviceSynchronize();

    // scan block sums again
    scan_block_3 <<<1, 512>>> (bb_sums, bb_sums_scanned, bb_len, dummy);
    cudaDeviceSynchronize();

    // apply inner blocks
    apply_block_sum <<<blocks_padded/1024, 512>>> (block_sums_scanned, blocks_intermediate, bb_len, bb_sums_scanned);
    cudaDeviceSynchronize();
    cudaEventRecord(end, stream);

    // apply block sums to intermediate
    apply_block_sum <<<z_values/1024, 512>>> (intermediate, h_output, values_p, blocks_intermediate);
    cudaDeviceSynchronize();
    cudaEventRecord(end, stream);

    /* 

    PERFORM NECESSARY DATA TRANSFER HERE

    */

    cudaStreamSynchronize(stream);
    for (int i=0; i<15; i++){
        printf("data: %i\n", data[i]);
    }
    for (int i=61435; i<61451; i++){
        printf("out: %i\n", h_output[i]);
    }
    printf("first: %i\n", h_output[0]);
    printf("last: %i\n", h_output[values-1]);
    printf("last: %i\n", h_output[z_values-1]);
    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    //DEALLOCATE RESOURCES HERE
    cudaFree(data_p);
    cudaFree(values_p);
    cudaFree(intermediate);
    cudaFree(block_sums);
    cudaFree(block_sums_scanned);
    cudaFree(dummy);
//    cudaFree(h_output);

//    for (int i=0; i<values; i++){
//        printf("val: %i\n", h_output[i]);
//    }

    int * reference_output = serial_implementation(data, values);
    printf("reference last: %i\n", reference_output[values-1]);

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
//    free(h_output);
    cudaFree(h_output);

    return 0;
}
