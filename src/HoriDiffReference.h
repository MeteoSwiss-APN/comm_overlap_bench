/*
 * HoriDiffReference.h
 *
 *  Created on: Apr 14, 2014
 *      Author: carlosos
 */

#pragma once
#include "HoriDiffRepository.h"

class HoriDiffReference {
    DISALLOW_COPY_AND_ASSIGN(HoriDiffReference);
public:
    HoriDiffReference() {}
    void Init(HoriDiffRepository& repository) { pRepository_ = &repository; }

    void Generate();
private:
    HoriDiffRepository* pRepository_;

};

