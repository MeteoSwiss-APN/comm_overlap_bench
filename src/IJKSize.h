#pragma once
#include "Definitions.h"

class IJKSize {
public:
    constexpr IJKSize(int isize, int jsize, int ksize):
        isize(isize), jsize(jsize), ksize(ksize)
    {

    }

    constexpr IJKSize(const IJKSize& other):
        isize(other.isize), jsize(other.jsize), ksize(other.ksize)
    {

    }

    constexpr bool operator== (const IJKSize& other) const {
        return isize == other.isize && jsize == other.jsize && ksize == other.ksize;
    }

public:
    const int isize;
    const int jsize;
    const int ksize;

    constexpr int isizeFull() const
    {
        return isize+2*cNumBoundaryLines;
    }

    constexpr int jsizeFull() const
    {
        return jsize+2*cNumBoundaryLines;
    }

    constexpr int ksizeFull() const
    {
        return ksize;
    }
};
