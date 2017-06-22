#pragma once
#include "Definitions.h"
#include "IJKSize.h"
#include <memory>
#include <vector>

#include "SimpleStorage.h"
#include "SimpleSwappableStorage.h"

class Repository {
    DISALLOW_COPY_AND_ASSIGN(Repository);

  public:
    Repository(const IJKSize& domain);
    ~Repository();

    const IJKSize domain;

    SimpleSwappableStorage< Real >& field(size_t i) {
        return *fields_[i];
    }
    SimpleStorage< Real >& in(size_t i) {
        return field(i).in;
    }
    SimpleStorage< Real >& out(size_t i) {
        return field(i).out;
    }
    void swap();

  private:
    std::vector< std::unique_ptr< SimpleSwappableStorage< Real > > > fields_;
};
