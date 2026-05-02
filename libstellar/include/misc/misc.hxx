#include "misc/misc.hh"

namespace stellar::misc
{
    inline void log_error(std::ostream& os,
                          std::string_view file,
                          const int line,
                          std::string_view func,
                          std::string_view msg)
    {
        os << "[STELLAR][ERROR] [" << file << ":" << line << "] in " << func << ": "
           << msg << std::endl;
        throw std::runtime_error{""};
    }
} // namespace stellar::misc