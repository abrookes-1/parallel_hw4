#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

/*

WRITE CUDA KERNEL FOR COUNT HERE

*/

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
    int * data = (int *)malloc(sizeof(int) * values);

    // Generate "random" vector
    std::mt19937 gen(13); // Keep constant to maintain determinism between runs
    std::uniform_int_distribution<> dist(0, 50);
    for (int i = 0; i < values; i++) {
        data[i] = dist(gen);
    }

    cudaStream_t stream;
    cudaEvent_t begin, end;
    cudaStreamCreate(&stream);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    int * h_output = (int *)malloc(sizeof(int) * values); // THIS VARIABLE SHOULD HOLD THE TOTAL COUNT BY THE END

    /*

    PERFORM NECESSARY VARIABLE DECLARATIONS HERE

    PERFORM NECESSARY DATA TRANSFER HERE

    */

    cudaEventRecord(begin, stream);

    /*

    LAUNCH KERNEL HERE

    */

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

    return 0;
}
