#pragma once

#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "VerificationFramework.h"
#include "HoriDiffRepository.h"
#include "CommunicationConfiguration.h"

class UnittestEnvironment
{
private:
    UnittestEnvironment() : pRepository_(0)
    {
        SetUp();
    }

    UnittestEnvironment(const UnittestEnvironment&): pRepository_(0)
    {
    }

public:
    static UnittestEnvironment& getInstance();

    void SetUp();
    virtual void TearDown();

    HoriDiffRepository& repository() { return *pRepository_; }
    IJKSize calculationDomain() { return calculationDomain_;}

    CommunicationConfiguration& communicationConfiguration() { return communicationConfiguration_; }

private:
    //calculation domain
    IJKSize calculationDomain_;

    // store the repository as pointer
    // could not properly free CUDA memory in a static destructor
    HoriDiffRepository* pRepository_;
    // halo update configuration
    CommunicationConfiguration communicationConfiguration_;

};

  
