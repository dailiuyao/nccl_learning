# nccl_learning

The first project called "ncclp2p" has two processes, each of which controls one GPU. These two processes will communicate with each other using ncclSend() and ncclRecv() for peer-to-peer (P2P) communication. When the project is run, the two processes will be located on two separate nodes.
