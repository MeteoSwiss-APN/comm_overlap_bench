#pragma once

#include <string>
#include <map>
#include <iostream>
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>

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

    int argc_;
    char** argv_;
public: 
    static Options& getInstance();

    static void setCommandLineParameters(int argc, char** argv) {
        getInstance().argc_ = argc;
        getInstance().argv_ = argv;
    }

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

    static void parse(const std::string& option, const std::string& argument, bool default_value)
    {
        int argc = Options::getInstance().argc_;
        char** argv = Options::getInstance().argv_;

        bool result = default_value;
        for(int i = 0; i < argc; ++i)
        {
            if(argv[i] == argument)
            {
                result = true;
                break;
            }
        }
        Options::set(option, result);
    }

    static void parse(const std::string& option, const std::string& argument, std::string default_value)
    {
        int argc = Options::getInstance().argc_;
        char** argv = Options::getInstance().argv_;

        std::string result = default_value;
        for(int i = 0; i < argc; ++i)
        {
            if(argv[i] == argument && i+1 < argc)
            {
                result = argv[i+1];
                break;
            }
        }
        Options::set(option, result);
    }

    template<typename T>
    static void parse(const std::string& option, const std::string& argument, T default_value);

};



template<typename T>
void Options::parse(const std::string& option, const std::string& argument, T default_value)
{
    int argc = Options::getInstance().argc_;
    char** argv = Options::getInstance().argv_;

    int result = default_value;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == argument && i+1 < argc)
        {
            result = boost::lexical_cast<T>(argv[i+1]);
            break;
        }
    }
    Options::set(option, result);
}
