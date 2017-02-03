#pragma once

class IJKSize {
public:
    IJKSize(int isize, int jsize, int ksize):
        isize(isize), jsize(jsize), ksize(ksize)
    {

    }

    IJKSize(const IJKSize& other):
        isize(other.isize), jsize(other.jsize), ksize(other.ksize)
    {

    }

    bool operator== (const IJKSize& other) const {
        return isize == other.isize && jsize == other.jsize && ksize == other.ksize;
    }

public:
    const int isize;
    const int jsize;
    const int ksize;
};
