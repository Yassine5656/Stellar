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

#ifndef INIT_HH
#define INIT_HH

#include "core/tensor/fwd.hh"

namespace stellar::core::tensor::init
{
    /**
     * @param tensor Tensor
     * @param scalar New value
     * @brief Set all tensor elements to 'scalar'.
    */
    void fill_constant(Tensor& tensor, scalar_t scalar);

    /**
     * @param tensor Tensor
     * @param scalar New value
     * @brief Fill the main diagonal by 'scalar'.
    */
    void fill_diagonal(Tensor& tensor, scalar_t scalar);

    /**
     * @param tensor Tensor
     * @brief Transform tensor to identity.
     * Thus, the main diagonal is set to 1, and other elements to 0.
    */
    void fill_identity(Tensor& tensor);

    /**
     * @param tensor Tensor
     * @param scalar value to be set
     * @param upper_section Determines the section filled.
     * If true, the upper section is filled.
     * @brief Change in-place to diagonal tensor.
    */
    void fill_triangular(Tensor& tensor, scalar_t scalar, bool upper_section);

    /**
     * @param tensor Tensor
     * @param begin First sequence element
     * @param end Last sequence element
     * @brief Fill tensor by an increasing sequence from begin to end
    */
    void fill_sequence(Tensor& tensor, int begin, unsigned int end);

    /**
     * @param tensor Tensor
     * @param min Min range
     * @param max Max range
     * @param seed Seed random generator
     * @brief Fill tensor with random values followed by uniform distribution.
    */
    void fill_uniform(Tensor& tensor, scalar_t min, scalar_t max, unsigned int seed);

    /**
     * @param tensor Tensor
     * @param mean Mean
     * @param stddev Standard deviation
     * @param seed Seed random generator
     * @brief Fill tensor with random values followed by normal distribution.
    */
    void fill_normal(Tensor& tensor, scalar_t mean, scalar_t stddev, unsigned int seed);

    /**
     * @param tensor Tensor
     * @param seed Seed random generator
     * @brief Fill tensor with random values followed by uniform distribution and xavier limit.
    */
    void fill_xavier_uniform(Tensor& tensor, unsigned int seed);

    /**
     * @param tensor Tensor
     * @param seed Seed random generator
     * @brief Fill tensor with random values followed by normal distribution and he standard deviation.
    */
    void fill_he_normal(Tensor& tensor, unsigned int seed);

} // namespace stellar::core::tensor::init

#endif // ! INIT_HH