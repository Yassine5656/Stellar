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

#include "core/tensor/ops.hh"

#include <limits>

#include "core/tensor/tensor.hh"

namespace stellar::core::tensor::ops
{
    Tensor add(const Tensor& a, const Tensor& b)
    {
        // Different dimensions
        if (a.rows() != b.rows() || a.cols() != b.cols())
        {
            STELLAR_ERROR("addition forbidden\n");
        }
        // Out-place tensor result
        Tensor result{a.size(), a.rows(), a.cols()};

        auto a_data      = a.unsafe_data();
        auto b_data      = b.unsafe_data();
        auto result_data = result.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(a, a_data, b_data, result_data) if (a.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < a.rows(); ++row)
        {
            for (unsigned int col = 0; col < a.cols(); ++col)
            {
                const unsigned int index = row * a.cols() + col;
                result_data[index]       = a_data[index] + b_data[index];
            }
        }

        return result;
    }

    void add_(Tensor& a, const Tensor& b)
    {
        // Different dimensions
        if (a.rows() != b.rows() || a.cols() != b.cols())
        {
            STELLAR_ERROR("addition forbidden\n");
        }

        auto a_data = a.unsafe_data();
        auto b_data = b.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(a, a_data, b_data) if (a.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < a.rows(); ++row)
        {
            for (unsigned int col = 0; col < a.cols(); ++col)
            {
                const unsigned int index = row * a.cols() + col;
                a_data[index] += b_data[index];
            }
        }
    }

    Tensor sub(const Tensor& a, const Tensor& b)
    {
        // Different dimensions
        if (a.rows() != b.rows() || a.cols() != b.cols())
        {
            STELLAR_ERROR("substraction forbidden\n");
        }

        // Out-place tensor result
        Tensor result{a.size(), a.rows(), a.cols()};

        auto a_data      = a.unsafe_data();
        auto b_data      = b.unsafe_data();
        auto result_data = result.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(a, a_data, b_data, result_data) if (a.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < a.rows(); ++row)
        {
            for (unsigned int col = 0; col < a.cols(); ++col)
            {
                const unsigned int index = row * a.cols() + col;
                result_data[index]       = a_data[index] - b_data[index];
            }
        }

        return result;
    }

    void sub_(Tensor& a, const Tensor& b)
    {
        // Different dimensions
        if (a.rows() != b.rows() || a.cols() != b.cols())
        {
            STELLAR_ERROR("substraction forbidden\n");
        }

        auto a_data = a.unsafe_data();
        auto b_data = b.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(a, a_data, b_data) if (a.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < a.rows(); ++row)
        {
            for (unsigned int col = 0; col < a.cols(); ++col)
            {
                const unsigned int index = row * a.cols() + col;
                a_data[index] -= b_data[index];
            }
        }
    }

    Tensor multiply(const Tensor& a, const Tensor& b)
    {
        if (a.cols() != b.rows())
        {
            STELLAR_ERROR("multiplication forbidden\n");
        }
        // Out-place tensor result
        Tensor result{a.rows(), b.cols()};

        auto a_data      = a.unsafe_data();
        auto b_data      = b.unsafe_data();
        auto result_data = result.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(a, b, a_data, b_data, result_data) if (a.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < a.rows(); ++row)
        {
            for (unsigned int k = 0; k < a.cols(); ++k)
            {
                const scalar_t temp = a_data[row * a.cols() + k];
                for (unsigned int col = 0; col < b.cols(); ++col)
                {
                    result_data[row * b.cols() + col] +=
                      temp * b_data[k * b.cols() + col];
                }
            }
        }

        return result;
    }

    Tensor multiplyScalar(const Tensor& tensor, scalar_t scalar)
    {
        Tensor result{tensor.size(), tensor.rows(), tensor.cols()};

        auto a_data      = tensor.unsafe_data();
        auto result_data = result.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(result_data, a_data, tensor, scalar) if (tensor.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < tensor.rows(); ++row)
        {
            for (unsigned int col = 0; col < tensor.cols(); ++col)
            {
                const unsigned int index = row * tensor.cols() + col;
                result_data[index]       = scalar * a_data[index];
            }
        }

        return result;
    }

    void multiplyScalar_(Tensor& tensor, scalar_t scalar)
    {
        auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(tensor, tensor_data, scalar) if (tensor.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < tensor.rows(); ++row)
        {
            for (unsigned int col = 0; col < tensor.cols(); ++col)
            {
                const unsigned int index = row * tensor.cols() + col;
                tensor_data[index] *= scalar;
            }
        }
    }

    scalar_t sum(const Tensor& tensor)
    {
        scalar_t sum = 0.0f;

        const auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(tensor, tensor_data, sum) if (tensor.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < tensor.rows(); ++row)
        {
            for (unsigned int col = 0; col < tensor.cols(); ++col)
            {
                sum += tensor_data[row * tensor.cols() + col];
            }
        }

        return sum;
    }

    scalar_t mean(const Tensor& tensor)
    {
        scalar_t sum = 0.0f;

        const auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel for default(none) shared(tensor, tensor_data, sum) if (tensor.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        for (unsigned int row = 0; row < tensor.rows(); ++row)
        {
            for (unsigned int col = 0; col < tensor.cols(); ++col)
            {
                sum += tensor_data[row * tensor.cols() + col];
            }
        }

        return sum / tensor.size();
    }

    scalar_t max(const Tensor& tensor)
    {
        scalar_t max = std::numeric_limits<scalar_t>::lowest();

        const auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel default(none) shared(tensor, tensor_data, max) if (tensor.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        {
            scalar_t local_max = std::numeric_limits<scalar_t>::lowest();

            // clang-format off
            #pragma omp for nowait
            // clang-format on
            for (unsigned int row = 0; row < tensor.rows(); ++row)
            {
                for (unsigned int col = 0; col < tensor.cols(); ++col)
                {
                    const unsigned int index = row * tensor.cols() + col;
                    if (local_max < tensor_data[index])
                    {
                        local_max = tensor_data[index];
                    }
                }
            }

            // clang-format off
            #pragma omp critical
            // clang-format on
            {
                if (local_max > max)
                {
                    max = local_max;
                }
            }
        }

        return max;
    }

    scalar_t min(const Tensor& tensor)
    {
        scalar_t min = std::numeric_limits<scalar_t>::max();

        const auto tensor_data = tensor.unsafe_data();

        // clang-format off
        #pragma omp parallel default(none) shared(tensor, tensor_data, min) if (tensor.rows() >= PARALLEL_THRESHOLD)
        // clang-format on
        {
            scalar_t local_min = std::numeric_limits<scalar_t>::max();

            // clang-format off
            #pragma omp for nowait
            // clang-format on
            for (unsigned int row = 0; row < tensor.rows(); ++row)
            {
                for (unsigned int col = 0; col < tensor.cols(); ++col)
                {
                    const unsigned int index = row * tensor.cols() + col;
                    if (tensor_data[index] < local_min)
                    {
                        local_min = tensor_data[index];
                    }
                }
            }

            // clang-format off
            #pragma omp critical
            // clang-format on
            {
                if (local_min < min)
                {
                    min = local_min;
                }
            }
        }

        return min;
    }
} // namespace stellar::core::tensor::ops