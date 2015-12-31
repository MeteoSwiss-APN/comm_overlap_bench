/*
 * HoriDiffRepository.h
 *
 *  Created on: Apr 14, 2014
 *      Author: carlosos
 */

#pragma once

#include <map>
#include <set>
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "SharedInfrastructure.h"
#include "Definitions.h"

/**
* @class HoriDiffRepository
* The class instantiates and owns all the global data fields used in the benchmark
* for the Horizontal Diffusion stencils
*
* Replacement for DycoreRepository, with only the required fields for the benchmark
*/
class HoriDiffRepository
{
    DISALLOW_COPY_AND_ASSIGN(HoriDiffRepository);
    BOOST_STATIC_ASSERT(N_HORIDIFF_VARS > 0);
public:

    HoriDiffRepository();
    ~HoriDiffRepository();

    /**
    * Method initializing the repository
    * @param calculationDomain calculation domain size
    *   only sets domain size - allocates no memory
    */
    void Init(const IJKSize& calculationDomain);


    /**
    * Method initializing the dycore repository and allocating the memory of repository data fields
    * (used by the unit test usually the data fields are allocated by the dycore wrapper repository)
    * @param calculationDomain calculation domain size
    */
    void AllocateDataFields();

    /**
    * Method for benchmarking that sets initial values for the fields that are reasonable
    */
    void SetInitalValues();

    /**
    * @return calculation domain size
    */
    const IJKSize& calculationDomain() const { return calculationDomain_; }

    IJKRealField& u_in(int idx) {
        assert(idx < N_HORIDIFF_VARS);
        assert(horiDiffFields_[idx]);
        return horiDiffFields_[idx]->in();
    }
    IJKRealField& u_out(int idx) {
        assert(idx < N_HORIDIFF_VARS);
        assert(horiDiffFields_[idx]);
        return horiDiffFields_[idx]->out();
    }

    SwapDataField<IJKRealField>& u(const int idx)
    {
        assert(idx < N_HORIDIFF_VARS);
        return *(horiDiffFields_[idx]);
    }

    void Swap()
    {
        for(size_t i=0; i < N_HORIDIFF_VARS; ++i)
        {
            u(i).Swap();
        }
    }

    std::vector< boost::shared_ptr< SwapDataField<IJKRealField> > >& horiDiffFields() { return horiDiffFields_; }

    // external parameter fields
    JRealField& crlat0() { return crlat0_; };
    JRealField& crlat1() { return crlat1_; };
    JRealField& crlato() { return crlato_; };
    JRealField& crlatu() { return crlatu_; };
    IJKRealField& hdmaskvel() { return hdmaskvel_; }
    IJKRealField& refField() {return refField_; }
    IJKRealField& ref4St() {return ref4St_; }

private:
    // configuration of the dynamics
    IJKSize calculationDomain_;

    // prognostic variables

    std::vector<boost::shared_ptr< SwapDataField<IJKRealField> > > horiDiffFields_;

    IJKRealField refField_;
    IJKRealField ref4St_;

    // external parameter fields
    JRealField crlat0_; // cosine of transformed latitude
    JRealField crlat1_; // cosine of transformed latitude
    JRealField crlato_;
    JRealField crlatu_;
    IJKRealField hdmaskvel_;
};

