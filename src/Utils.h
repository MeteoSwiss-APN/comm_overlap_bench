#pragma once

    template<typename TField>
    void SetFieldToValue(TField& f, typename TField::ValueType val)
    {
        IJKBoundary const &b = f.boundary();
        IJKSize const &s = f.calculationDomain();
        int Jb = b.jMinusOffset();
        int Je = s.jSize()+b.jPlusOffset();
        for( int k=b.kMinusOffset(); k<s.kSize()+b.kPlusOffset(); k++ ){
            for( int j=Jb; j<Je; j++ ){
                for( int i=b.iMinusOffset(); i<s.iSize()+b.iPlusOffset(); i++ ){
                    f(i,j,k) = val;
                }
            }
        }
    }
