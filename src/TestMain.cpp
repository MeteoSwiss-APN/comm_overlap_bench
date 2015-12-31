#include <algorithm>
#include <boost/lexical_cast.hpp>
#include "gtest/gtest.h"
#include "Options.h"
#include "UnittestEnvironment.h"

// method parsing a string option
int parseIntOption(int argc, char **argv, std::string option, int defaultValue)
{
    int result = defaultValue;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == option && i+1 < argc)
        {
            result = boost::lexical_cast<int>(argv[i+1]);
            break;
        }
    }
    return result;
}

// method parsing a string option
std::string parseStringOption(int argc, char **argv, std::string option, std::string defaultValue)
{
    std::string result = defaultValue;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == option && i+1 < argc)
        {
            result = argv[i+1];
            break;
        }
    }
    return result;
}

// method parsing a boolean option
bool parseBoolOption(int argc, char **argv, std::string option)
{
    bool result = false;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == option)
        {
            result = true;
            break;
        }
    }
    return result;
}

// override main in order to read further program arguments
int main(int argc, char **argv) 
{
    std::cout << "HP2CDycoreUnittest\n\n";
    std::cout << "usage: HP2CDycoreUnittest -p <dataPath>" << "\n";
    for(int i=0; i < argc; ++i)
        std::cout << std::string(argv[i]) << std::endl;
    // parse google test options
    ::testing::InitGoogleTest(&argc, argv);



    // parse additional command options
    int iSize = parseIntOption(argc, argv, "--ie", 128);
    int jSize = parseIntOption(argc, argv, "--je", 128);
    int kSize = parseIntOption(argc, argv, "--ke", 60);
    bool sync = parseBoolOption(argc, argv, "--sync");

    IJKSize domain;
    domain.Init(iSize, jSize, kSize);
    Options::getInstance().domain_ = domain;
    Options::getInstance().sync_ = sync;

    // register environment
    testing::AddGlobalTestEnvironment(&UnittestEnvironment::getInstance());

    return RUN_ALL_TESTS();
}

  
