#include <iostream>
#include <fstream>
#include <cstdio>
#include <random>

void read_sparse_file(char * name, float ** data, int ** ptr, int ** indices, int * nz, int * row, int * col) {
    
    FILE *fp;
    char line[1024];
    char * dump;

    if ((fp = fopen(name, "r")) == NULL) {
        abort();
    }

    dump = fgets(line, 128, fp);
    if (dump == nullptr) abort();
    
    while (line[0] == '%') {
        dump = fgets(line, 128, fp);
        if (dump == nullptr) abort();
    }
    
    sscanf(line, "%d %d %d\n", row, col, nz);
    *ptr = (int *)malloc((*row + 1) * sizeof(int));
    *indices = (int *)malloc(*nz * sizeof(int));
    *data = (float *)malloc(*nz * sizeof(float));

    int lastr = 0;
    for (int i = 0; i < *nz; i++) {
        int r;
        int check = fscanf(fp, "%d %d %f\n", &r, *indices + i, *data + i);
        if (check != 3) abort();
        (*indices)[i]--;
        if (r != lastr) {
            (*ptr)[r-1] = i;
            lastr = r;
        }
    }
    (*ptr)[*row] = *nz;
}

