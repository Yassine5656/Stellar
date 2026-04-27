#ifndef MISC_H
#define MISC_H

#include <ostream>

namespace stellar::misc
{
    void log_error(std::ostream& os,
                   std::string_view file,
                   const int line,
                   std::string_view func,
                   std::string_view msg);
} // namespace stellar::misc

#include "misc/misc.hxx"

#endif // ! MISC_H