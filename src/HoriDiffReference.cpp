/*
 * HoriDiffReference.cpp
 *
 *  Created on: Apr 14, 2014
 *      Author: carlosos
 */
#include "HoriDiffReference.h"

void HoriDiffReference::Generate() {

    assert(pRepository_->horiDiffFields()[0]);
    //all the input fields in the vector are supposed to contain the same source data
    //We just take the first one in order to generate the reference output data
    IJKRealField& sourceField = pRepository_->u_in(0);
    IJKRealField& refField = pRepository_->refField();
    JRealField& crlato = pRepository_->crlato();
    JRealField& crlatu = pRepository_->crlatu();
    IJKRealField& hdmaskvel = pRepository_->hdmaskvel();
    const IJKSize domain = pRepository_->calculationDomain();
    const IJKBoundary boundary = refField.boundary();
    IJRealField lap, flux_x, flux_y;
    KBoundary kBoundary;
    kBoundary.Init(0,0);
    lap.Init("lap", domain, kBoundary);
    flux_x.Init("flux_x", domain, kBoundary);
    flux_y.Init("flux_y", domain, kBoundary);

    #pragma omp for nowait
    for(int k=boundary.kMinusOffset(); k < domain.kSize() + boundary.kPlusOffset(); ++k)
    {
        for(int i=0; i < domain.iSize(); ++i) {
            for(int j=0; j < domain.jSize(); ++j)
            {
                refField(i,j,k) = sourceField(i-1,j,k) + sourceField(i+1,j,k) + sourceField(i,j-1,k) + sourceField(i,j+1,k) - 4.0*sourceField(i,j,k);
            }
        }
/*        for(int i=-1; i < domain.iSize()+1; ++i) {
            for(int j=-1; j < domain.jSize()+1; ++j)
            {
                double fieldFlux = sourceField(i+1,j,k) - sourceField(i,j,k);
                flux_x(i,j,k) = lap(i+1,j,k) - lap(i,j,k);
                if(flux_x(i,j,k) * (fieldFlux) > 0) {
                    flux_x(i,j,k) = 0;
                }
                fieldFlux = sourceField(i,j+1,k) - sourceField(i,j,k);
                flux_y(i,j,k) = crlato(i,j,k)*(lap(i,j+1,k) - lap(i,j,k));
                if(flux_y(i,j,k) * (fieldFlux) > 0) {
                    flux_y(i,j,k) = 0;
                }
            }
        }
        for(int i=0; i < domain.iSize(); ++i) {
            for(int j=0; j < domain.jSize(); ++j)
            {
                refField(i,j,k) = sourceField(i,j,k) + (flux_x(i-1,j,k) - flux_x(i,j,k) +
                        flux_y(i,j-1,k) - flux_y(i,j,k))*hdmaskvel(i,j,k);
            }
        }
*/
    }

}


