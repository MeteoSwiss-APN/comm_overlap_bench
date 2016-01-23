#pragma once

#include <string>
#include "SharedInfrastructure.h"
/**
* @class Options
* Singleton data container for program options
*/
class Options /* singleton */
{ 
private: 
    Options() { }
    Options(const Options& other) { }
    ~Options() { }
public: 
    static Options& getInstance(); 

    IJKSize domain_;
    bool sync_;
    bool nocomm_;
    bool nocomp_;
    bool nostella_;
    bool nogcl_;
    int nHaloUpdates_;
    int nRep_;
    bool inOrder_;
}; 

  
