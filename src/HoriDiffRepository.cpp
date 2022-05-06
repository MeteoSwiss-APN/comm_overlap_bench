/*
 * HoriDiffRepository.cpp
 *
 *  Created on: Apr 14, 2014
 *      Author: carlosos
 */
#include "HoriDiffRepository.h"
#include "Definitions.h"
#include "Utils.h"
#include <cmath>
#include <memory>
#include <sstream>

HoriDiffRepository::HoriDiffRepository() {}

HoriDiffRepository::~HoriDiffRepository() {}

void HoriDiffRepository::Init(const IJKSize& calculationDomain) {
    // set the calculation domain
    calculationDomain_ = calculationDomain;

    for (size_t i = 0; i < N_HORIDIFF_VARS; ++i) {
        horiDiffFields_.push_back(std::make_shared< SwapDataField< IJKRealField > >());
    }
}

void HoriDiffRepository::AllocateDataFields() {
    // setup diffusion data fields
    KBoundary boundary;
    // full size fields
    boundary.Init(0, 0);
    hdmaskvel_.Init("hdmaskvel", calculationDomain_, boundary);
    crlato_.Init("crlato", calculationDomain_, boundary);
    crlatu_.Init("crlatu", calculationDomain_, boundary);

    crlat0_.Init("crlat0", calculationDomain_, boundary);
    crlat1_.Init("crlat1", calculationDomain_, boundary);

    refField_.Init("HoriDiffRefField", calculationDomain_, boundary);
    ref4St_.Init("ref4St", calculationDomain_, boundary);

    for (size_t i = 0; i < horiDiffFields_.size(); ++i) {
        std::stringstream ss;
        ss << i;
        horiDiffFields_[i]->Init(std::string("horiField") + ss.str(), calculationDomain_, boundary);
    }
}

void HoriDiffRepository::SetInitalValues() {
    for (size_t i = 0; i < N_HORIDIFF_VARS; ++i) {
        IJKRealField& u_in = (horiDiffFields_[i])->in();
        IJKRealField& u_out = (horiDiffFields_[i])->out();

        // set the fields to diffuse
        // values are selected that are in the range of those from the Performance unit test

        IJKBoundary const& b = u_in.boundary();
        IJKSize const& s = u_in.calculationDomain();
        int i1 = b.iMinusOffset();
        int i2 = s.iSize() + b.iPlusOffset();
        int j1 = b.jMinusOffset();
        int j2 = s.jSize() + b.jPlusOffset();
        double dx = 1. / (double)(i2 - i1);
        double dy = 1. / (double)(j2 - j1);

        // u values between 5 and 13
        for (int j = j1; j < j2; j++) {
            for (int i = i1; i < i2; i++) {
                double x = dx * (double)i;
                double y = dy * (double)j;
                for (int k = b.kMinusOffset(); k < s.kSize() + b.kPlusOffset(); k++) {
                    u_in(i, j, k) = 5. + 8. * (2. + cos(PI * (x + 1.5 * y)) + sin(2 * PI * (x + 1.5 * y))) / 4.;
                    u_out(i, j, k) = 0;
                }
            }
        }
    }

    // sample ranges for crlat0 and crlat1, taken from 64x64 mesh:
    // we will interpolate linearly between these to get our input values:
    // crlat0: 0.994954 to 0.995156
    // crlat0 = [0.994954 0.994994 0.995035 0.995076 0.995116 0.995156 0.995195 0.995232 0.995267 0.9953 0.995331
    // 0.995359 0.995384 0.995406 0.995425 0.99544 0.995451 0.995459 0.995462 0.995462 0.995459 0.995451 0.99544
    // 0.995425 0.995406 0.995384 0.995359 0.995331 0.9953 0.995267 0.995232 0.995195 0.995156 0.995116 0.995076
    // 0.995035 0.994994 0.994954 0.994914 0.994876 0.994839 0.994805 0.994772 0.994743 0.994716 0.994692 0.994672
    // 0.994656 0.994644 0.994636 0.994631 0.994631 0.994636 0.994644 0.994656 0.994672 0.994692 0.994716 0.994743
    // 0.994772 0.994805 0.994839 0.994876 0.994914 0.994954 0.994994 0.995035 0.995076 0.995116 0.995156];
    // crlat1: 0.994924 to 0.995143
    // crlat1 = [0.994924 0.994967 0.995011 0.995056 0.9951 0.995143 0.995186 0.995227 0.995266 0.995303 0.995337
    // 0.995369 0.995398 0.995423 0.995445 0.995463 0.995477 0.995488 0.995494 0.995496 0.995494 0.995488 0.995477
    // 0.995463 0.995445 0.995423 0.995398 0.995369 0.995337 0.995303 0.995266 0.995227 0.995186 0.995143 0.9951
    // 0.995056 0.995011 0.994967 0.994924 0.994882 0.994841 0.994802 0.994766 0.994732 0.994701 0.994674 0.99465
    // 0.99463 0.994615 0.994604 0.994597 0.994595 0.994597 0.994604 0.994615 0.99463 0.99465 0.994674 0.994701 0.994732
    // 0.994766 0.994802 0.994841 0.994882 0.994924 0.994967 0.995011 0.995056 0.9951 0.995143];
    IJKBoundary const& cb = crlat0().boundary();
    int nj = calculationDomain_.jSize() + cb.jPlusOffset() - cb.jMinusOffset();
    double delta0 = (0.995156 - 0.994954) / (double)(nj - 1);
    double delta1 = (0.995143 - 0.994924) / (double)(nj - 1);
    for (int j = cb.jMinusOffset(); j < calculationDomain_.jSize() + cb.jPlusOffset(); j++) {
        crlat0()(0, j, 0) = 0.994954 + (double)(j - cb.jMinusOffset()) * delta0;
        crlat1()(0, j, 0) = 0.994924 + (double)(j - cb.jMinusOffset()) * delta1;
    }

    // setup 1d fields
    for (int j = 1 - cNumBoundaryLines; j < calculationDomain_.jSize() + (cNumBoundaryLines - 1); ++j) {
        crlato_(0, j, 0) = crlat1()(0, j, 0) / crlat0()(0, j, 0);
        crlatu_(0, j, 0) = crlat1()(0, j - 1, 0) / crlat0()(0, j, 0);
    }
    crlato_(0, -cNumBoundaryLines, 0) = crlat1()(0, -cNumBoundaryLines, 0) / crlat0()(0, -cNumBoundaryLines, 0);

    SetFieldToValue(hdmaskvel_, 0.025);
}
