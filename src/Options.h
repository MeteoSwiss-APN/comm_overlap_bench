#pragma once

#include <string>
#include <map>
#include <boost/any.hpp>
/**
* @class Options
* Singleton data container for program options
*/
class Options /* singleton */
{ 
private: 
    Options()
    { }
    ~Options() { }

    std::map<std::string, boost::any> options_;
public: 
    static Options& getInstance();

    template<typename T>
    static void set(const std::string& name, const T& value)
    {
        getInstance().options_[name] = value;
    }
    static void set(const std::string& name, const char* value)
    {
        getInstance().options_[name] = std::string(value);
    }

    template<typename T>
    static const T& get(const std::string& name) {
        return *boost::any_cast<T>(&(getInstance().options_[name]));
    }
}; 

  
