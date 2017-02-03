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

    void swapWith(SimpleStorage<T>& other);
    const std::string   name;
    const IJKSize       size;
};

template <typename T>
SimpleStorage<T>::SimpleStorage(const IJKSize& size, const std::string& name):
    name(name), size(size)
{
    const size_t s = size.isize*size.jsize*size.ksize;
    host = (T*) malloc(s*sizeof(T));
    cudaMalloc(reinterpret_cast<void**>(&device), s*sizeof(T));
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
