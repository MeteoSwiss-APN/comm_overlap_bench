#pragma once

/**
* @struct Laplacian
* Returns the Laplacian (lap) of an input array (s)
*/
template<typename TEnv>
struct Laplacian
{
    STENCIL_FUNCTION(TEnv)

    FUNCTION_PARAMETER(0, data) // input data field
    FUNCTION_PARAMETER(1, crlatvo) // cosine ratio at j+1/2 for metrics
    FUNCTION_PARAMETER(2, crlatvu) // cosine ratio at j-1/2 for metrics

    __ACC__
    static T Do(Context ctx)
    {
        return  ctx[data::At(iplus1)] + ctx[data::At(iminus1)] + ctx[data::At(jplus1)] + ctx[data::At(jminus1)] - (T)4.0 * ctx[data::Center()];
    }
};

