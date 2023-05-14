#include "cuda_runtime.h"
#include "nccl.h"

typedef struct {
    ncclComm_t ncclComm;
} MyNcclComm_t;


void my_ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, MyNcclComm_t comm, cudaStream_t stream);


