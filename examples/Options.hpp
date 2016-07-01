/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
