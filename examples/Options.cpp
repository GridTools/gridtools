#include "Options.hpp"


Options& Options::getInstance() 
{
    static Options instance; 
    return instance; 
}

  
