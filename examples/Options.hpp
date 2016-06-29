#pragma once

#include <string>

/**
* @class Options
* Singleton data container for program options
*/
class Options /* singleton */
    {
  private:
    Options() {
        for (int i = 0; i < 4; ++i) {
            m_size[i] = 0;
        }
        m_verify = true;
    }
    Options(const Options &) {}
    ~Options() {}

  public:
    static Options &getInstance();

    int m_size[4];
    bool m_verify;
};
