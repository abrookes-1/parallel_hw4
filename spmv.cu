#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "util.h"

constexpr float THRESHOLD = 1e-5f;
//constexpr float THRESHOLD = 1e-6f;
const int threads_per_row = 16;
/*

WRITE CUDA KERNEL FOR COUNT HERE

*/

float * serial_implementation(float * sparse_matrix, int * ptr, int * indices, float * dense_vector, int rows) {
    float * output = (float *)malloc(sizeof(float) * rows);
    
    for (int i = 0; i < rows; i++) {
        float accumulator = 0.f;
        for (int j = ptr[i]; j < ptr[i+1]; j++) {
            accumulator += sparse_matrix[j] * dense_vector[indices[j]];
        }
        output[i] = accumulator;
    }
    
    return output;
}

__global__ void sparse_partial(float * sparse_matrix, int * ptr, int * indices, float * dense_vector, float * out, int * test){
    __shared__ float partial_sums[threads_per_row];
    int row = blockIdx.x;
    int x = threadIdx.x; // 0-15
    int row_len = ptr[row+1] - ptr[row];
    //int my_len = row_len / 16;
    int thread_len = (row_len + threads_per_row - 1) / threads_per_row;
    float my_partial = 0;

    for (int i=0; i<thread_len; i++){
        int row_indx = x * thread_len + i;
        int sparse_indx = ptr[row] + row_indx;
        if (row_indx < row_len){
            my_partial += sparse_matrix[sparse_indx] * dense_vector[indices[sparse_indx]];
        }
    }

    partial_sums[x] = my_partial;
    __syncthreads();

    // reduce
    // TODO: use proper reduce
    if (x == 0) {
        for (int i=1; i<threads_per_row; i++){
            partial_sums[0] += partial_sums[i];
        }
        out[row] = partial_sums[0];
    }
}


int main(int argc, char ** argv) {
    
    assert(argc == 2);
    
    float * sparse_matrix = nullptr; 
    float * dense_vector = nullptr;
    
    int * ptr = nullptr;
    int * indices = nullptr;
    int values = 0, rows = 0, cols = 0;

    int *test;
    cudaMallocManaged(&sparse_matrix, values * sizeof(float));
    cudaMallocManaged(&ptr, (rows+1) * sizeof(int));
    cudaMallocManaged(&indices, values * sizeof(int));
    cudaMallocManaged(&dense_vector, cols * sizeof(float));
    cudaMallocManaged(&test, sizeof(int));

    read_sparse_file(argv[1], &sparse_matrix, &ptr, &indices, &values, &rows, &cols);
    printf("%d %d %d\n", values, rows, cols);
    dense_vector = (float *)malloc(sizeof(float) * cols);

    // Generate "random" vector
    std::mt19937 gen(13); // Keep constant to maintain determinism between runs
    std::uniform_real_distribution<> dist(-10.0f, 10.0f);
    for (int i = 0; i < cols; i++) {
        dense_vector[i] = dist(gen);
    }

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    float * h_output = (float *)malloc(sizeof(float) * rows); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END
    cudaMallocManaged(&h_output, rows * sizeof(float));
    /*

    PERFORM NECESSARY VARIABLE DECLARATIONS HERE

    PERFORM NECESSARY DATA TRANSFER HERE

    */
    float *sparse_matrix_p, *dense_vector_p;
    int *ptr_p, *indices_p;
    cudaMallocManaged(&sparse_matrix_p, values * sizeof(float));
    cudaMallocManaged(&ptr_p, (rows+1) * sizeof(int));
    cudaMallocManaged(&indices_p, values * sizeof(int));
    cudaMallocManaged(&dense_vector_p, cols * sizeof(float));

    for (int i=0; i<values; i++){
        sparse_matrix_p[i] = sparse_matrix[i];
        indices_p[i] = indices[i];
    }
    for (int i=0; i<rows+1; i++){
        ptr_p[i] = ptr[i];
    }
    for (int i=0; i<cols; i++){
        dense_vector_p[i] = dense_vector[i];
    }
//    for (int i=0; i<5; i++) {
//        printf("dense_vector: %f", dense_vector[i]);
//        printf("\n");
//    }
//    for (int i=0; i<5; i++) {
//        printf("ptr: %i", ptr[i]);
//        printf("\n");
//    }

    dim3 grid_dim(rows, 1, 1);
    dim3 block_dim(16, 1, 1);

    cudaEventRecord(begin, stream);

    /*

    LAUNCH KERNEL HERE

    */
    *test = 29.0;
    sparse_partial <<<grid_dim, block_dim>>> (sparse_matrix_p, ptr_p, indices_p, dense_vector_p, h_output, test);
    cudaDeviceSynchronize();

    cudaEventRecord(end, stream);

    /* 

    PERFORM NECESSARY DATA TRANSFER HERE

    */

    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, begin, end);
    printf("Elapsed time: %f ms\n", ms);

    /* 

    DEALLOCATE RESOURCES HERE

    */
    cudaFree(sparse_matrix_p);
    cudaFree(ptr_p);
    cudaFree(indices_p);
    cudaFree(dense_vector_p);
    cudaFree(h_output);

//    for (int i=0; i<5; i++) {
//        printf("out: %f", h_output[i]);
//        printf("\n");
//    }

    float * reference_output = serial_implementation(sparse_matrix, ptr, indices, dense_vector, rows);
    for (int i = 0; i < rows; i++) {
        if (fabs(reference_output[i] - h_output[i]) > THRESHOLD) {
            printf("ERROR: %f != %f at index %d\n", reference_output[i], h_output[i], i);
            abort();
        }
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);

    free(sparse_matrix);
    free(dense_vector);
    free(ptr);
    free(indices);
    free(reference_output);
//    free(h_output); // throws invalid pointer

    return 0;
}
