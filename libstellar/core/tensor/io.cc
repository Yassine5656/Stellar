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

#include "core/tensor/io.hh"

#include <array>
#include <cstring>
#include <fstream>

#include "core/tensor/tensor.hh"

constexpr std::string_view MAGIC_NUMBER{"stellar"};

namespace stellar::core::tensor::io
{
    namespace
    {
        inline void WRITE_STELLAR_HEADER(std::ostream& os, const Tensor& tensor)
        {
            os << MAGIC_NUMBER << '\n'
               << tensor.getRows() << '\n'
               << tensor.getCols() << '\n';
        }

        bool check_magic_number(std::ifstream& os)
        {
            std::string line;
            std::getline(os, line);

            return line == MAGIC_NUMBER;
        }

        std::array<char, sizeof(scalar_t)> scalar_t_to_bytes(const scalar_t scalar)
        {
            std::array<char, sizeof(scalar_t)> res;
            std::memcpy(res.data(), &scalar, sizeof(scalar_t));
            return res;
        }
    } // namespace

    void save(Tensor& tensor, const std::string& filename)
    {
        std::ofstream file{filename, std::ios::binary};
        if (!file.is_open())
        {
            STELLAR_ERROR("File doesn't exist. filename: '" + filename + "'\n");
        }
        // Write stellar-format header
        WRITE_STELLAR_HEADER(file, tensor);

        auto tensor_data = tensor.unsafe_data();

        for (unsigned int row = 0; row < tensor.getRows(); ++row)
        {
            for (unsigned int column = 0; column < tensor.getCols(); ++column)
            {
                std::array<char, sizeof(scalar_t)> binary_weight;
                binary_weight =
                  scalar_t_to_bytes(tensor_data[row * tensor.getCols() + column]);
                for (unsigned int i = 0; i < sizeof(scalar_t); ++i)
                {
                    file << binary_weight[i];
                }
            }
        }

        file.close();
    }

    Tensor load(const std::string& filename)
    {
        std::ifstream file{filename, std::ios::binary};
        if (!file.is_open())
        {
            STELLAR_ERROR("File does not exist! Filename: '" + filename + "'\n");
        }
        if (!check_magic_number(file))
        {
            file.close();
            STELLAR_ERROR("Invalid Magic Number!\n");
        }
        std::string line;
        std::getline(file, line);
        const unsigned int rows = std::strtoul(line.data(), nullptr, 10);
        std::getline(file, line);
        const unsigned int cols = std::strtoul(line.data(), nullptr, 10);

        // Out-place Tensor result
        Tensor result{rows, cols};
        auto result_data = result.unsafe_data();

        for (unsigned int row = 0; row < rows; ++row)
        {
            for (unsigned int col = 0; col < cols; ++col)
            {
                std::array<char, sizeof(scalar_t)> arr;
                if (!file.read(arr.data(), sizeof(scalar_t)))
                {
                    file.close();
                    STELLAR_ERROR("Premature End of File at row " + std::to_string(row)
                                  + "\n");
                }

                scalar_t scalar;
                std::memcpy(&scalar, arr.data(), sizeof(scalar_t));
                result_data[row * result.getCols() + col] = scalar;
            }
        }
        if (std::string dummy; file.read(dummy.data(), 1))
        {
            file.close();
            STELLAR_ERROR("File is larger than expected (trailing characters)!\n");
        }

        file.close();
        return result;
    }

    void load_(Tensor& tensor, const std::string& filename)
    {
        std::ifstream file{filename, std::ios::binary};
        if (!file.is_open())
        {
            STELLAR_ERROR("File does not exist! Filename: '" + filename + "'\n");
        }
        if (!check_magic_number(file))
        {
            file.close();
            STELLAR_ERROR("Invalid Magic Number!\n");
        }
        std::string line;
        std::getline(file, line);
        const unsigned int rows = std::strtoul(line.data(), nullptr, 10);
        std::getline(file, line);
        const unsigned int cols = std::strtoul(line.data(), nullptr, 10);

        auto tensor_data = tensor.unsafe_data();

        for (unsigned int row = 0; row < rows; ++row)
        {
            for (unsigned int col = 0; col < cols; ++col)
            {
                std::array<char, sizeof(scalar_t)> arr;
                if (!file.read(arr.data(), sizeof(scalar_t)))
                {
                    file.close();
                    STELLAR_ERROR("Premature End of File at row " + std::to_string(row)
                                  + "\n");
                }

                scalar_t scalar;
                std::memcpy(&scalar, arr.data(), sizeof(scalar_t));
                tensor_data[row * tensor.getCols() + col] = scalar;
            }
        }
        if (std::string dummy; file.read(dummy.data(), 1))
        {
            file.close();
            STELLAR_ERROR("File is larger than expected (trailing characters)!\n");
        }

        file.close();
    }

} // namespace stellar::core::tensor::io