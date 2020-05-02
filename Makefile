CXX := nvcc
OPT_FLAGS := -O3
DEBUG_FLAGS := -G
GENCODE := -gencode arch=compute_70,code=compute_70 -gencode arch=compute_75,code=compute_75

.phony: clean all debug release

all: debug release

release: spmv_opt scan_opt

debug: spmv_debug scan_debug

clean:
	rm -f scan_opt scan_debug spmv_opt spmv_debug

spmv_opt: spmv.cu util.h
	$(CXX) $(OPT_FLAGS) -o $@ $< $(GENCODE)

scan_opt: scan.cu
	$(CXX) $(OPT_FLAGS) -o $@ $< $(GENCODE)

spmv_debug: spmv.cu util.h
	$(CXX) $(DEBUG_FLAGS) -o $@ $< $(GENCODE)

scan_debug: scan.cu
	$(CXX) $(DEBUG_FLAGS) -o $@ $< $(GENCODE)

handin.tar: scan.cu spmv.cu
	tar -cvf handin.tar scan.cu spmv.cu

.phony: clean

