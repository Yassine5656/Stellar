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

#include "core/tensor/misc.hh"

#include <iomanip>

#include "core/tensor/ops.hh"
#include "core/tensor/tensor.hh"

constexpr unsigned int MAX_ELM_RENDERED   = 3;
constexpr unsigned int PADDING            = 10;
constexpr unsigned int PRECISION_RENDERED = 4;

namespace stellar::core::tensor::misc
{
    void dump(const Tensor& tensor)
    {
        const auto backup_flags = std::cout.flags();

        auto tensor_data = tensor.unsafe_data();

        std::cout << std::setprecision(PRECISION_RENDERED);
        // Dump dimensions
        std::cout << "Tensor: [" << tensor.getRows() << " x " << tensor.getCols() << "]"
                  << "\n";
        const unsigned int rows_rendered = std::min(MAX_ELM_RENDERED, tensor.getRows());
        const unsigned int cols_rendered = std::min(MAX_ELM_RENDERED, tensor.getCols());

        // Dump tensor
        std::cout << "[" << "\n";
        for (unsigned int row = 0; row < rows_rendered; ++row)
        {
            std::cout << std::setw(PADDING) << " " << tensor_data[row * tensor.getCols()];
            unsigned int col = 1;
            for (; col < cols_rendered; ++col)
            {
                std::cout << ", " << std::setw(PADDING)
                          << tensor_data[row * tensor.getCols() + col];
            }
            if (tensor.getCols() - cols_rendered > col)
            {
                std::cout << ", " << std::setw(PADDING) << "...";
            }
            for (col = std::max(col, tensor.getCols() - cols_rendered);
                 col < tensor.getCols(); ++col)
            {
                std::cout << ", " << std::setw(PADDING)
                          << tensor_data[row * tensor.getCols() + col];
            }
            std::cout << "\n";
        }

        if (tensor.getRows() - rows_rendered > rows_rendered)
        {
            std::cout << " " << std::setw(PADDING) << "...";
            unsigned int col = 1;
            for (; col < cols_rendered; ++col)
            {
                std::cout << ", " << std::setw(PADDING) << "...";
            }
            if (tensor.getCols() - cols_rendered > col)
            {
                std::cout << ", " << std::setw(PADDING) << "...";
            }
            for (col = std::max(col, tensor.getCols() - cols_rendered);
                 col < tensor.getCols(); ++col)
            {
                std::cout << ", " << std::setw(PADDING) << "...";
            }
            std::cout << "\n";
        }

        for (unsigned int row = std::max(rows_rendered, tensor.getRows() - rows_rendered);
             row < tensor.getRows(); ++row)
        {
            std::cout << std::setw(PADDING) << " " << tensor_data[row * tensor.getCols()];
            unsigned int col = 1;
            for (; col < cols_rendered; ++col)
            {
                std::cout << ", " << std::setw(PADDING)
                          << tensor_data[row * tensor.getCols() + col];
            }
            if (tensor.getCols() - cols_rendered > col)
            {
                std::cout << ", " << std::setw(PADDING) << "...";
            }
            for (col = std::max(col, tensor.getCols() - cols_rendered);
                 col < tensor.getCols(); ++col)
            {
                std::cout << ", " << std::setw(PADDING)
                          << tensor_data[row * tensor.getCols() + col];
            }
            std::cout << "\n";
        }
        std::cout << "]" << "\n";

        // Dump tensor's characteristics
        std::cout << "Min: " << ops::min(tensor) << "\n";
        std::cout << "Max: " << ops::max(tensor) << "\n";
        std::cout << "Mean: " << ops::mean(tensor) << "\n";

        // Restore cout flags
        std::cout.flags(backup_flags);
    }

} // namespace stellar::core::tensor::misc