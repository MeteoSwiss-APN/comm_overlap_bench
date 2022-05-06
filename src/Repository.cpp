#include "Repository.h"

Repository::Repository(const IJKSize& domain)
    : domain(domain) {
    fields_.reserve(N_HORIDIFF_VARS);

    for (size_t i = 0; i < N_HORIDIFF_VARS; ++i) {
        auto f = new SimpleSwappableStorage< Real >(domain, "field" + std::to_string(i));
        fields_.push_back(std::unique_ptr< SimpleSwappableStorage< Real > >(f));
    }
}

Repository::~Repository() {}

void Repository::swap() {
    for (auto& f : fields_) {
        f->swap();
    }
}
