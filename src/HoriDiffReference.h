/*
 * HoriDiffReference.h
 *
 *  Created on: Apr 14, 2014
 *      Author: carlosos
 */

#pragma once
#include "Repository.h"
#include <memory>
//#include "HoriDiffRepository.h"

class HoriDiffReference {
    DISALLOW_COPY_AND_ASSIGN(HoriDiffReference);
public:
    HoriDiffReference(std::shared_ptr<Repository> repository):
        repository_(repository)
    {}

    void Generate();
private:
    std::shared_ptr<Repository> repository_;
};

