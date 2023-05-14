module load cuda

module load mpich/3.4.2-nvidiahpc-21.9-0

export MPI_HOME=$(which mpirun | sed 's/\/bin\/mpirun//g')
export PATH="${MPI_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="${MPI_HOME}/include:$C_INCLUDE_PATH"

export NCCL_HOME=/home/ldai8/NCCL/deps-nccl/nccl/build
export C_INCLUDE_PATH="${NCCL_HOME}/include:$C_INCLUDE_PATH"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:$LD_LIBRARY_PATH"

nvcc -I/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0/include ncclp2p.cc -o ncclp2p  -lnccl -lmpi
