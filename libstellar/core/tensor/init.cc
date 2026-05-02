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

#include "core/tensor/init.hh"

#include <random>
#include <omp.h>

#include "core/tensor//tensor.hh"

namespace stellar::core::tensor::init
{
    void fill_constant(Tensor& tensor, scalar_t scalar)
    {
        auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(tensor, tensor_data, scalar) if (tensor.getRows() >= PARALLEL_THRESHOLD || tensor.getCols() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int i = 0; i < tensor.getRows() * tensor.getCols(); ++i)
            tensor_data[i] = scalar;
    }

    void fill_diagonal(Tensor& tensor, scalar_t scalar)
    {
        // Assert square tensor
        if (tensor.getRows() != tensor.getCols())
        {
            STELLAR_ERROR("Tensor is not square one.\n");
        }

        auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(tensor, tensor_data, scalar) if (tensor.getRows() >= PARALLEL_THRESHOLD || tensor.getCols() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < tensor.getRows(); ++row)
        {
            for (unsigned col = 0; col < row; ++col)
                tensor_data[row * tensor.getCols() + col] = 0;
            for (unsigned col = row + 1; col < tensor.getCols(); ++col)
                tensor_data[row * tensor.getCols() + col] = 0;
            tensor_data[row * tensor.getCols() + row] = scalar;
        }
    }

    void fill_identity(Tensor& tensor) { fill_diagonal(tensor, 1); }

    void fill_triangular(Tensor& tensor, scalar_t scalar, bool upper_section)
    {
        // Assert square matrix
        if (tensor.getRows() != tensor.getCols())
        {
            STELLAR_ERROR("Tensor is not square one.\n");
        }
        unsigned int end_row = tensor.getRows();
        if (upper_section)
        {
            ++end_row;
        }

        auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel default(none) shared(tensor, tensor_data, scalar, upper_section, end_row) if (tensor.getRows() >= PARALLEL_THRESHOLD || tensor.getCols() >= PARALLEL_THRESHOLD)
        // clang-format on
        {
            // clang-format off
            #pragma omp for
            // clang-format on
            for (unsigned int row = 0; row < tensor.getRows(); ++row)
            {
                for (unsigned int col = 0; col < row; ++col)
                {
                    tensor_data[row * tensor.getCols() + col] = 0;
                }
                unsigned int col = row;
                if (!upper_section)
                {
                    ++col;
                }
                for (; col < tensor.getRows(); ++col)
                {
                    tensor_data[row * tensor.getCols() + col] = scalar;
                }
            }
        }
    }

    void fill_sequence(Tensor& tensor, int begin, unsigned int end)
    {
        // Assert increasing sequence
        if (begin > static_cast<int>(end))
        {
            STELLAR_ERROR("Tensor is not square one.\n");
        }

        auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(tensor, tensor_data, begin, end) if (tensor.getCols() >= PARALLEL_THRESHOLD || tensor.getRows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int index = 0; index < tensor.getSize(); ++index)
        {
            tensor_data[index] =
              static_cast<scalar_t>(begin + static_cast<int>(index % (end + 1)));
        }
    }

    void fill_uniform(Tensor& tensor, scalar_t min, scalar_t max, unsigned int seed)
    {
        auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel default(none) shared(tensor, tensor_data, min, max, seed) if (tensor.getCols() >= PARALLEL_THRESHOLD || tensor.getRows() >= PARALLEL_THRESHOLD)
        // clang-format on
        {
            const int thread_id = omp_get_thread_num();
            auto generator      = std::mt19937(seed + thread_id);
            auto distribution   = std::uniform_real_distribution(min, max);
            // clang-format off
            #pragma omp for
            // clang-format on
            for (unsigned int index = 0; index < tensor.getSize(); ++index)
            {
                tensor_data[index] = distribution(generator);
            }
        }
    }

    void fill_normal(Tensor& tensor, scalar_t mean, scalar_t stddev, unsigned int seed)
    {
        auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel default(none) shared(tensor, tensor_data, mean, stddev, seed) if (tensor.getCols() >= PARALLEL_THRESHOLD || tensor.getRows() >= PARALLEL_THRESHOLD)
        // clang-format on
        {
            const int thread_id = omp_get_thread_num();
            auto generator      = std::mt19937(seed + thread_id);
            auto distribution   = std::normal_distribution(mean, stddev);
            // clang-format off
            #pragma omp for
            // clang-format on
            for (unsigned int index = 0; index < tensor.getSize(); ++index)
            {
                tensor_data[index] = distribution(generator);
            }
        }
    }

    void fill_xavier_uniform(Tensor& tensor, unsigned int seed)
    {
        auto tensor_data = tensor.unsafe_data();

        scalar_t xavier_scalar =
          std::sqrt(6.0f / static_cast<scalar_t>(tensor.getRows() + tensor.getCols()));
        // clang-format off
        #pragma omp parallel default(none) shared(tensor, tensor_data, xavier_scalar, seed) if (tensor.getCols() >= PARALLEL_THRESHOLD || tensor.getRows() >= PARALLEL_THRESHOLD)
        // clang-format on
        {
            const int thread_id = omp_get_thread_num();
            auto generator      = std::mt19937(seed + thread_id);
            auto distribution =
              std::uniform_real_distribution(-xavier_scalar, xavier_scalar);
            // clang-format off
            #pragma omp for
            // clang-format on
            for (unsigned int index = 0; index < tensor.getSize(); ++index)
            {
                tensor_data[index] = distribution(generator);
            }
        }
    }

    void fill_he_normal(Tensor& tensor, unsigned int seed)
    {
        auto tensor_data = tensor.unsafe_data();

        scalar_t he_scalar = std::sqrt(2.0f / static_cast<scalar_t>(tensor.getCols()));
        // clang-format off
        #pragma omp parallel default(none) shared(tensor, tensor_data, he_scalar, seed) if (tensor.getCols() >= PARALLEL_THRESHOLD || tensor.getRows() >= PARALLEL_THRESHOLD)
        // clang-format on
        {
            const int thread_id = omp_get_thread_num();
            auto generator      = std::mt19937(seed + thread_id);
            auto distribution   = std::normal_distribution<scalar_t>(0, he_scalar);
            // clang-format off
            #pragma omp for
            // clang-format on
            for (unsigned int index = 0; index < tensor.getSize(); ++index)
            {
                tensor_data[index] = distribution(generator);
            }
        }
    }
} // namespace stellar::core::tensor::init