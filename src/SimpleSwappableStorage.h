#pragma once
#include "SimpleStorage.h"

template <typename T>
class SimpleSwappableStorage {
    DISALLOW_COPY_AND_ASSIGN(SimpleSwappableStorage);
public:
    SimpleSwappableStorage(const IJKSize& size, const std::string& name);
    ~SimpleSwappableStorage();

    void swap();

    SimpleStorage<Real> in;
    SimpleStorage<Real> out;

    const std::string   name;
    const IJKSize       size;
};

template <typename T>
SimpleSwappableStorage<T>::SimpleSwappableStorage(const IJKSize &size, const std::string &name):
    in(size, name+"_in"),
    out(size, name+"_out"),
    name(name),
    size(size)
{

}

template <typename T>
SimpleSwappableStorage<T>::~SimpleSwappableStorage()
{

}

template <typename T>
void SimpleSwappableStorage<T>::swap() {
    in.swapWith(out);
}
