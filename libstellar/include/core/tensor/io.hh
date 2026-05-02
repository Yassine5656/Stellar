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

#ifndef IO_HH
#define IO_HH

#include <string>

#include "core/tensor/fwd.hh"

namespace stellar::core::tensor::io
{
    /**
     * @param filename Path of the binary file
     * @brief Create a tensor from the binary file
    */
    Tensor load(const std::string& filename);

    /**
     * @param tensor Tensor
     * @param filename Path of the binary file
     * @brief Load the tensor from the binary file
    */
    Tensor load_(Tensor& tensor, const std::string& filename);

    /**
     * @param tensor Tensor to save
     * @param filename Path of the binary file
     * @brief Save the tensor in the binary file
    */
    Tensor save(Tensor& tensor, const std::string& filename);
} // namespace stellar::core::tensor::io

#endif // ! IO_HH