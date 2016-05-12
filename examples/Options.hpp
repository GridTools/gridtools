#pragma once

#include <string>

/**
* @class Options
* Singleton data container for program options
*/
class Options /* singleton */
{ 
private: 
    Options() { }
    Options(const Options&) { }
    ~Options() { }
public: 
    static Options& getInstance(); 

    int m_size[3];
}; 

  
