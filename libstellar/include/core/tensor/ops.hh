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

#ifndef OPS_HH
#define OPS_HH

#include "core/tensor/fwd.hh"

namespace stellar::core::tensor::ops
{
    /**
     * @brief Tensor out-place addition
    */
    Tensor add(const Tensor& a, const Tensor& b);

    /**
     * @param a Tensor result
     * @brief Tensor in-place addition
    */
    void add_(Tensor& a, const Tensor& b);

    /**
     * @brief Tensor out-place substraction
    */
    Tensor sub(const Tensor& a, const Tensor& b);

    /**
     * @param a Tensor result
     * @brief Tensor in-place substraction
    */
    void sub_(Tensor& a, const Tensor& b);

    /**
     * @brief Tensor out-place product
    */
    Tensor multiply(const Tensor& a, const Tensor& b);

    /**
     * @param a Tensor
     * @param scalar Scalar coefficient
     * @brief Tensor out-place scalar product
    */
    Tensor multiplyScalar(const Tensor& a, scalar_t scalar);

    /**
     * @param a Tensor
     * @param scalar Scalar coefficient
     * @brief Tensor in-place scalar product
    */
    void multiplyScalar_(Tensor& a, scalar_t scalar);

    /**
     * @brief Sum of tensor elements
     */
    scalar_t sum(const Tensor& tensor);

    /**
     * @brief Mean of tensor elements
     */
    scalar_t mean(const Tensor& tensor);

    /**
     * @brief Max of tensor elements
     */
    scalar_t max(const Tensor& tensor);

    /**
     * @brief Min of tensor elements
     */
    scalar_t min(const Tensor& tensor);

} // namespace stellar::core::tensor::ops

#endif // ! OPS_HH