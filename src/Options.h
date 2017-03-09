#pragma once

#include <string>
#include <map>
#include <iostream>
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

    std::map<std::string, int> optionsInt_;
    std::map<std::string, bool> optionsBool_;
    std::map<std::string, std::string> optionsString_;

    int argc_;
    char** argv_;
public: 
    static Options& getInstance();

    static void setCommandLineParameters(int argc, char** argv) {
        getInstance().argc_ = argc;
        getInstance().argv_ = argv;
    }

    static void set(const std::string& name, bool value)
    {
        getInstance().optionsBool_[name] = value;
    }
    static void set(const std::string& name, int value)
    {
        getInstance().optionsInt_[name] = value;
    }
    static void set(const std::string &name, const std::string& value)
    {
        getInstance().optionsString_[name] = value;
    }

    static void set(const std::string& name, const char* value)
    {
         std::string val(value);
         set(name, val);
    }

    static const bool& getBool(const std::string &name)
    {
        return getInstance().optionsBool_[name];
    }

    static const int& getInt(const std::string &name)
    {
        return getInstance().optionsInt_[name];
    }

    static const std::string& getString(const std::string& name)
    {
        return getInstance().optionsString_[name];
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

    static void parse(const std::string& option, const std::string& argument, int default_value)
    {
        int argc = Options::getInstance().argc_;
        char** argv = Options::getInstance().argv_;

        int result = default_value;
        for(int i = 0; i < argc; ++i)
        {
            if(argv[i] == argument && i+1 < argc)
            {
                result = std::stoi(std::string(argv[i+1]));
                break;
            }
        }
        Options::set(option, result);
    }

};
