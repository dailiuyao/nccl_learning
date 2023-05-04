#!/bin/bash
#SBATCH --nodes=2
##SBATCH --nodelist=gnode002,gnode003,gnode005,gnode008
#SBATCH --partition gpu 
#SBATCH --qos=gpu-ext
#SBATCH --time=0-00:29:00 
#SBATCH --ntasks-per-node=56 
#SBATCH --output=ncclp2p-1nodes_%j.stdout    
#SBATCH --job-name=ncclp2p-1nodes    
#SBATCH --gres=gpu:2

# ---[ Script Setup ]---

set -e

# ---[ Set Up cuda/nccl/nccl-test/mpi ]---
module load cuda
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda"
export CUDA_HOME
# export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda/lib64:$LD_LIBRARY_PATH

module load mpich/3.4.2-nvidiahpc-21.9-0
export MPI_HOME=$(which mpirun | sed 's/\/bin\/mpirun//g')
export PATH="${MPI_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:$LD_LIBRARY_PATH"

export NCCL_HOME=/home/ldai8/NCCL/deps-nccl/nccl/build
# export NCCL_TEST_HOME=/home/ldai8/NCCL/deps-nccl/nccl-tests/build
export PATH="${NCCL_HOME}/include:$PATH"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:$LD_LIBRARY_PATH"

echo "########## ENVIRONMENT ########"
echo "NCCL_LOCATION=${NCCL_HOME}"
# echo "NCCL_TEST_LOCATION=${NCCL_TEST_HOME}"
# echo " ### nccl based nccl-test ###"
# echo " ### how to check ###"
# ldd /home/ldai8/NCCL/deps-nccl/nccl-tests/build/all_reduce_perf
# echo " ### check end ###"
echo ""

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Tree
# export NCCL_PROTO=LL128

pushd /home/ldai8/nccl-learning/p2p

nvcc -I/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0/include ncclp2p.cc -o ncclp2p -L/home/ldai8/NCCL/deps-nccl/nccl/build/lib -lnccl -L/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0/lib -lmpi

set -x
mpirun -np 2 -ppn 1 /home/ldai8/nccl-learning/p2p/ncclp2p

set +x

popd