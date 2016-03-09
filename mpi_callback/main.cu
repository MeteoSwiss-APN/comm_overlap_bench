#include "mpi.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include<cuda.h>
#include <future>
#include <cassert>

#define cudaCall(val) __checkCudaErrors__ ( (val), #val, __FILE__, __LINE__ )
 
/*
A test program that tries to call mpi_isend from within a cuda event callback.
Each rank starts S streams and sends them asynchronously to its right neighbor
in a ring. Each rank first initializes the send buffers, then issues an mpi_irecv for a device buffer,
then calls a memcpy h-to-d, followed by a kernel in a stream; the stream also enques a
callback that when executed starts the mpi_isend from a device buffer.
It then waits for everybody and copies the buffers back onto the host to print.

Author: Christoph Angerer
*/

#define USE_GPU
#define USE_CALLBACK
#define USE_BACKGROUND_ISEND
template <typename T>
inline void __checkCudaErrors__(T code, const char *func, const char *file, int line) 
{
    if (code) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, (unsigned int)code, func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

//keep #streams and #mpi ranks < 10 to keep the 1-digit encoding intact
#define S 8
#define N 100

__global__ void MyKernel(int myid, int *buffer)
{
    buffer[threadIdx.x] += 10*myid;
}

struct CallbackInfo
{
    int *send_buffer_d;
    int device_id;
    int dest;
    int tag;
    int myid;
    MPI_Request send_request;
};

void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data){
    CallbackInfo *info = (CallbackInfo*)data;
    printf("Callback called: dest=%d, tag=%d\n", info->dest, info->tag);
    CUdevice dev;
    int result = cuCtxGetDevice(&dev);
    printf("cuCtxGetDevice inside callback result=%d\n", result);
   
    printf("Using device_id %d\n", info->device_id);
#ifdef USE_BACKGROUND_ISEND
    auto future = std::async(std::launch::async, [&info]()
    { 
        //need to set the device, otherwise I get a "illegal context" error
        cudaCall(cudaSetDevice(info->device_id));
        printf("Hello from device %d tag %d\n", info->device_id, info->tag);
        CUdevice dev;
        int result = cuCtxGetDevice(&dev);
        printf("cuCtxGetDevice inside callback inside background thread result=%d\n", result);
        //MPI_Isend and MPI_send both deadlock here.
        printf("Sending %d %p %d %d %d\n", info->myid, info->send_buffer_d, info->dest, N, info->tag);
        MPI_Send(info->send_buffer_d, N, MPI_INT, info->dest, info->tag, MPI_COMM_WORLD); 
        printf("Bye %d %d %d\n", info->myid, info->dest, info->tag);
    });
#else
    //This is what we want, but it fails with a CUDA_ERROR_NOT_PERMITTED in cuCtxtGetDevice()
    MPI_Isend(info->send_buffer_d, N, MPI_INT, info->dest, info->tag, MPI_COMM_WORLD, &info->send_request);
#endif
}

int main(int argc, char *argv[])
{
    int myid, numprocs, left, right;
    int recv_buffer[S][N], send_buffer[S][N];
    CallbackInfo infos[S];

    MPI_Request recv_request[S];
    MPI_Status status;

    const char* myid_c = std::getenv("SLURM_PROCID");
    if(!myid_c) {
        printf("SLURM_PROCID not set");
        exit (EXIT_FAILURE);
    }
    const char* nprocs_c = std::getenv("SLURM_NPROCS");
    if(!nprocs_c) {
        printf( "SLURM_NPROCS not set");
        exit (EXIT_FAILURE);
    }

    const char* g2g_c = std::getenv("G2G");
    if(!g2g_c) {
        printf( "G2G not set");
        exit (EXIT_FAILURE);
    }

    myid = atoi(myid_c);
    numprocs = atoi(nprocs_c);
    int g2g = atoi(g2g_c);
    assert(g2g < 3 || g2g >= 0);

    int numgpus = numprocs;
    if(g2g!=2) 
        numgpus = 1;
    printf("NUMPROC %d %d\n", numgpus, myid % numgpus);
#ifdef USE_GPU
//    cudaCall(cudaGetDeviceCount(&numgpus));
//    printf("Rank %d uses device %d\n", myid, myid % numgpus);
    cudaCall(cudaSetDevice(myid % numgpus));
#endif

        printf("NUMPROC %d %d\n", numgpus, myid % numgpus);

    int provided;
    MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

//    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
//    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#ifdef USE_GPU
    int *recv_buffer_d[S];
    cudaStream_t streams[S];
#endif


    right = (myid + 1) % numprocs;
    left = myid - 1;
    if (left < 0)
        left = numprocs - 1;

#ifdef USE_GPU
    if(myid == 0) printf("\nUSING GPU!\n");
    #ifdef USE_CALLBACK
        if(myid==0) printf("USING CALLBACK!\n");
        #ifdef USE_BACKGROUND_ISEND
            if(myid==0) printf("With background MPI_ISEND\n\n");
        #else
            if(myid==0) printf("With direct MPI_ISEND\n\n");
        #endif
    #else
        if(myid==0) printf("USING NO CALLBACK\n\n");
    #endif
//    cudaCall(cudaGetDeviceCount(&numgpus));
//    printf("Rank %d uses device %d\n", myid, myid % numgpus);
//    cudaCall(cudaSetDevice(myid % numgpus));

    CUdevice dev;
    int result = cuCtxGetDevice(&dev);
    printf("cuCtxGetDevice outside callback result=%d; %d\n", result, myid);

    //create streams and device buffers
    for(int s = 0; s < S; s++) 
    {
        cudaCall(cudaStreamCreate(&streams[s]));
        cudaCall(cudaMalloc(&recv_buffer_d[s], N*sizeof(int)));
        cudaCall(cudaMalloc(&infos[s].send_buffer_d, N*sizeof(int)));
    }
#else
    if(myid==0) printf("\nUSING CPU!\n\n");
#endif

    //initialise send buffer elements with the stream number
    for(int s = 0; s < S; s++)
    {
        for(int i = 0; i < N; i++)
        {
            send_buffer[s][i] = s;
        }
    }

    if(myid == 1)
    {
        printf("Rank %d send buffer:\n", myid);
        printf("=========================================\n");
        for(int s = 0; s < S; s++)
        {
            for(int i = 0; i < N; i++)
            {
                printf("%2d,", send_buffer[s][i]);
            }
            printf("\n");
        }
    }
    for(int s = 0; s < S; s++) 
    {
        //kick off S receives on device
#ifdef USE_GPU
        MPI_Irecv(recv_buffer_d[s], N, MPI_INT, left, s, MPI_COMM_WORLD, &recv_request[s]);
#else
        MPI_Irecv(recv_buffer[s], N, MPI_INT, left, s, MPI_COMM_WORLD, &recv_request[s]);
#endif
        printf("IRECV %d from %d with tag %d\n", myid, left, s);
     
        printf("SETTING %d %d %d \n", myid,numgpus, myid % numgpus);
        infos[s].device_id = myid % numgpus;
        infos[s].dest = right;
        infos[s].tag = s;
        infos[s].myid = myid;
#ifdef USE_GPU
        //enqueue asyncronous memcpy and kernel
        cudaCall(cudaMemcpyAsync(infos[s].send_buffer_d, send_buffer[s], N*sizeof(int), cudaMemcpyHostToDevice, streams[s]));
        //the kernel will add 10*myid to the send_buffer so that the result is a number xy where x is id of the sender and y is the stream
        MyKernel<<<1,N,0,streams[s]>>>(myid, infos[s].send_buffer_d);

        printf("Kernel %d %d %d \n", myid, infos[s].device_id, numgpus);
    #ifdef USE_CALLBACK
        //enqueue the isend
        cudaCall(cudaStreamAddCallback(streams[s], MyCallback, &infos[s], 0));
    #else
        cudaCall(cudaStreamSynchronize(streams[s]));
        printf("Before ISend %d  to %d, size %d with tag %d \n", myid, infos[s].dest, N, infos[s].tag);

        MPI_Isend(infos[s].send_buffer_d, N, MPI_INT, infos[s].dest, infos[s].tag, MPI_COMM_WORLD, &infos[s].send_request);
    #endif

        printf("ISend %d \n", myid);
#else
        for(int i = 0; i < N; i++)
        {
            send_buffer[s][i] += 10*myid;
        }
        MPI_Isend(send_buffer[s], N, MPI_INT, right, s, MPI_COMM_WORLD, &infos[s].send_request);
#endif

    }
       
    for(int s = 0; s < S; s++)
    {
        printf("Waiting %d \n", myid);
        MPI_Wait(&recv_request[s], &status);
#ifndef USE_BACKGROUND_ISEND
        MPI_Wait(&infos[s].send_request, &status);
#endif
#ifdef USE_GPU
        cudaCall(cudaMemcpyAsync(recv_buffer[s], recv_buffer_d[s], N*sizeof(int), cudaMemcpyDeviceToHost, streams[s]));
#endif
    }
#ifdef USE_GPU
    cudaCall(cudaDeviceSynchronize());
#endif

    if(myid == 0)
    {
        printf("Rank %d got Result:\n", myid);
        printf("=========================================\n");
        for(int s = 0; s < S; s++)
        {
            for(int i = 0; i < N; i++)
            {
                //initialise send buffer elements with the stream number
                printf("%2d,", recv_buffer[s][i]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    printf("END %d \n", myid);
#ifdef USE_GPU
    for(int s = 0; s < S; s++)
    {
        cudaStreamDestroy(streams[s]);
        cudaCall(cudaFree(recv_buffer_d[s]));
        cudaCall(cudaFree(infos[s].send_buffer_d));
    }
#endif
    
    return 0;
}
