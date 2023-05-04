#include <stdio.h>
#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include <cstdlib>
#include <cassert>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[]){

    int datasize=10;

    // Initialize MPI
    MPI_Init(&argc, &argv);



    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check that we have exactly two ranks
    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: This example requires exactly 2 MPI ranks\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Initialize the NCCL communicator for this rank
    ncclUniqueId ncclId;
    ncclComm_t ncclComm;
    // Generate a unique NCCL ID on rank 0
    if (rank == 0) ncclGetUniqueId(&ncclId);
    MPICHECK(MPI_Bcast((void *)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        // Initialize the NCCLcommunicator for rank 0
        ncclCommInitRank(&ncclComm, 2, ncclId, 0);
    } else {
        // Initialize the NCCL communicator for rank 1
        ncclCommInitRank(&ncclComm, 2, ncclId, 1);
    }

    //allocating and initializing device buffers
    float* sendbuff;
    float* recvbuff;
    cudaStream_t s;

    CUDACHECK(cudaMalloc(&sendbuff, datasize * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, datasize * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    // Initialize the data on the GPU
    cudaMemset(sendbuff, 0, datasize * sizeof(float));

    float val[datasize];

    for (int i=0; i<datasize; i++){
        val[i] = i;
        printf("In rank %d, val[%d] is: %f\n", rank, i, val[i]);
    }

    cudaMemcpy(sendbuff, val, datasize * sizeof(float), cudaMemcpyHostToDevice);

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    // Send data from rank 0 to rank 1
    if(rank == 0)
    {
        ncclSend(sendbuff, datasize, ncclFloat, 1, ncclComm, s);
    }
    else
    {
        ncclRecv(recvbuff, datasize, ncclFloat, 0, ncclComm, s);
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation

    CUDACHECK(cudaStreamSynchronize(s));

    printf("this rank is: %d\n", rank);


    // Copy the data from the GPU to the CPU
    float* data_cpu = new float[datasize];
    cudaMemcpy(data_cpu, recvbuff, datasize * sizeof(float), cudaMemcpyDeviceToHost);

    if(rank == 1){
        for(int i=0; i<datasize; i++){
            assert(data_cpu[i] == (float)i);
        }
    }

    cudaFree(sendbuff);
    cudaFree(recvbuff);
    cudaStreamDestroy(s);
    ncclCommDestroy(ncclComm);

    MPI_Finalize();

    printf("rank %d Success \n", rank);

    return 0;
}