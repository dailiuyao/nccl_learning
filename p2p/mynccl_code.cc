#include "mynccl_code.h"

// struct MyNcclComm_t{
//                   ncclComm_t ncclComm;
// };

void my_ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, MyNcclComm_t comm, cudaStream_t stream){
                          ncclSend(sendbuff, count, datatype, peer, comm.ncclComm, stream);
}