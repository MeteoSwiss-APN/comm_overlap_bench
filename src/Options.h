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
    bool nostella_;
}; 

  
