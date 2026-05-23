// Copyright 2026 Yassine TLICH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MISC_HH
#define MISC_HH

#include <iostream>

#define STELLAR_ERROR(Msg)                                                               \
    stellar::misc::log_error(std::cerr, __FILE__, __LINE__, __func__, Msg)

namespace stellar::misc
{
    inline void log_error(std::ostream& os,
                          std::string_view file,
                          const int line,
                          std::string_view func,
                          std::string_view msg);
} // namespace stellar::misc

#include "misc/misc.hxx"

#endif // ! MISC_HH