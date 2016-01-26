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

    const ErrorMetric& metric() const { return metric_; }
    CommunicationConfiguration& communicationConfiguration() { return communicationConfiguration_; }

private:
    //calculation domain
    IJKSize calculationDomain_;

    // verification
    ErrorMetricLegacy metric_;

    // store the repository as pointer
    // could not properly free CUDA memory in a static destructor
    HoriDiffRepository* pRepository_;
    // halo update configuration
    CommunicationConfiguration communicationConfiguration_;

};

  
