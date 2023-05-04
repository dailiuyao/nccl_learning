#include <stdio.h>
#include "cuda_runtime.h"
#include <cstdlib>
#include <cassert>


int main(int argc, char* argv[])
{
int datasize = 10;
float data_cpu[datasize];
float* sendbuff;
cudaMalloc(&sendbuff, datasize * sizeof(float));

cudaMemset(sendbuff, 0, datasize * sizeof(float));

cudaMemcpy(data_cpu, sendbuff, data_size * sizeof(float), cudaMemcpyDeviceToHost);

for(int i=0; i<datasize; i++)
    {        
        printf("data_cpu[%d] is: %f\n", i, data_cpu[i]);
    }

  return 0;
}