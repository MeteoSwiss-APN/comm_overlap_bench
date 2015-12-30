#include <cassert>
#include <cmath>
#include <fstream>
#include "Options.h"
#include "Definitions.h"
#include "UnittestEnvironment.h"

UnittestEnvironment& UnittestEnvironment::getInstance()
{
    // google test deletes the instance, therefore only the new is done here
    static UnittestEnvironment* pInstance = new UnittestEnvironment();
    return *pInstance;
}

void UnittestEnvironment::SetUp()
{
    try
    {
        // make sure the repository is null
        assert(!pRepository_);
        pRepository_ = new HoriDiffRepository();

        // prepare the repository
        calculationDomain_.Init(
                Options::getInstance().domain_.iSize(),
                Options::getInstance().domain_.jSize(),
                Options::getInstance().domain_.kSize()
        );
        pRepository_->Init(calculationDomain_);
        pRepository_->AllocateDataFields();
        pRepository_->SetInitalValues();

        // verification objects
#ifdef SINGLEPRECISION
        metric_.Init(1e-6);
#else
        metric_.Init(1e-12);
#endif
        IJKSize globalDomainSize;
        globalDomainSize.Init(calculationDomain_.iSize() + cNumBoundaryLines*2,
            calculationDomain_.jSize() + cNumBoundaryLines*2,
            calculationDomain_.kSize());

        int subdomainPosition_[4];
        subdomainPosition_[0] = cNumBoundaryLines+1;
        subdomainPosition_[1] = cNumBoundaryLines+1;
        subdomainPosition_[2] = cNumBoundaryLines+calculationDomain_.iSize();
        subdomainPosition_[3] = cNumBoundaryLines+calculationDomain_.jSize();
        // Initialize the halo update configuration
        communicationConfiguration_.Init(true, true, false, false, false, false,
            globalDomainSize, 1, subdomainPosition_);
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error: testing environment not setup correctly:\n";
        std::cerr << ex.what() << "\n";
        ASSERT_TRUE(false);
    }
}


void UnittestEnvironment::TearDown()
{
    assert(pRepository_);
    delete pRepository_;
}
