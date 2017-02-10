#pragma once
#include <cuda_runtime.h>
#include <string>
#include <cstdlib>
#include "Definitions.h"
#include "IJKSize.h"

template <typename T>
class SimpleStorage {
    DISALLOW_COPY_AND_ASSIGN(SimpleStorage);
public:
    SimpleStorage(const IJKSize& size, const std::string& name);
    ~SimpleStorage();

    T* device;
    T* host;

    T& operator()(size_t i, size_t j, size_t k);

    void updateHost();
    void updateDevice();
    void swapWith(SimpleStorage<T>& other);

    const std::string   name;
    const IJKSize       size;
    const size_t        storage_size;
};

template <typename T>
SimpleStorage<T>::SimpleStorage(const IJKSize& size, const std::string& name):
    name(name),
    size(size),
    storage_size((size.isize+cNumBoundaryLines*2) * (size.jsize+cNumBoundaryLines*2) * size.ksize)
{
    host = (T*) malloc(storage_size*sizeof(T));
    cudaMalloc(reinterpret_cast<void**>(&device), storage_size*sizeof(T));
}

template <typename T>
SimpleStorage<T>::~SimpleStorage()
{
    delete host;
    cudaFree(device);
}

template <typename T>
void SimpleStorage<T>::swapWith(SimpleStorage<T> &other) {
    std::swap(device, other.device);
    std::swap(host, other.host);
}

template <typename T>
T& SimpleStorage<T>::operator ()(size_t i, size_t j, size_t k) {
    const size_t istride = 1;
    const size_t jstride = size.isize+cNumBoundaryLines*2;
    const size_t kstride = (size.jsize+cNumBoundaryLines*2)*jstride;
    return host[i*istride+j*jstride+k*kstride];
}

template <typename T>
void SimpleStorage<T>::updateDevice() {
    cudaMemcpy(device, host, storage_size, cudaMemcpyHostToDevice);
}

template <typename T>
void SimpleStorage<T>::updateHost() {
    cudaMemcpy(host, device, storage_size, cudaMemcpyDeviceToHost);
}
