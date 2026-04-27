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

#ifndef TENSOR_H
#define TENSOR_H

#include <cassert>
#include <string>
#include <vector>

namespace stellar::core
{
    using scalar_t = float;

    /**
     * @brief Tensor class used for deep learning purposes.
     */
    class Tensor
    {
      public:
        /**
         * @param size Number of rows
         * @param length Number of columns
         * @brief General tensor constructor
         */
        Tensor(const unsigned int size, const unsigned int length)
          : size_(size)
          , length_(length)
          , elements_(size_ * length_)
        {}

        /**
         * @param size Dimension
         * @brief Square tensor constructor
         */
        explicit Tensor(const unsigned int size)
          : size_(size)
          , length_(size)
          , elements_(size_ * length_)
        {}

        /**
         * @param other Matrix to move
         * @brief Move tensor constructor
         */
        Tensor(const Tensor& other) noexcept = default;

        Tensor& operator=(Tensor&&) noexcept = default;

        /**
         * @param other Matrix to copy
         * @param size Number of sub-tensor's rows
         * @param length Number of sub-tensor's columns
         * @param row_begin First row to copy
         * @param column_begin First column to copy
         * @brief Copy sub-tensor constructor
         */
        Tensor(const Tensor& other,
               unsigned int size,
               unsigned int length,
               unsigned int row_begin,
               unsigned int column_begin);

        /**
         * @brief Return the number of tensor's rows
         */
        [[nodiscard]] unsigned int getSize() const { return size_; }

        /**
         * @brief Return the number of tensor's columns
         */
        [[nodiscard]] unsigned int getLength() const { return length_; }

        /**
         * @param row Row index
         * @param col Column index
         * @brief Get the element of the row-th row and col-th column.
         * Indexes are zero-based.
         */
        [[nodiscard]] scalar_t get(const unsigned int row, const unsigned int col) const
        {
            assert(row < size_ && col < length_);
            return elements_[row * length_ + col];
        }

        /**
         * @param row Row index
         * @param col Column index
         * @param val New element
         * @brief Set a new element 'val' of the row-th row and col-th column.
         * Indexes are zero-based.
         */
        void set(const unsigned int row, const unsigned int col, const scalar_t val)
        {
            assert(row < size_ && col < length_);
            elements_[row * length_ + col] = val;
        }

        /**
         * @brief Tensor addition by parallelism of calculations and cache optimization.
         */
        void add(const Tensor& other);

        /**
         * @brief Tensor substraction by parallelism of calculations and cache optimization.
         */
        void sub(const Tensor& other);

        /**
         * @brief Tensor product by parallelism of calculations and cache optimization.
         */
        void multiply(const Tensor& other);

        /**
         * @param scalar Scalar coefficient
         * @brief Tensor scalar product
         */
        void multiplyScalar(scalar_t scalar);

        /**
         * @param scalar New value
         * @brief Set all tensor elements to 'scalar'.
         */
        void fill_constant(scalar_t scalar);

        /**
         * @param scalar New value
         * @brief Fill the main diagonal by 'scalar'.
         */
        void fill_diagonal(scalar_t scalar);

        /**
         * @brief Transform tensor to identity.
         * Thus, the main diagonal is set to 1, and other elements to 0.
         */
        void fill_identity() { fill_diagonal(1); }

        /**
         * @param scalar value to be set
         * @param upper_section Determines the section filled.
         * If true, the upper section is filled.
         * @brief Change in-place to diagonal tensor.
         */
        void fill_triangular(scalar_t scalar, bool upper_section);

        /**
         * @param begin First sequence element
         * @param end Last sequence element
         * @brief Fill tensor by an increasing sequence from begin to end
         */
        void fill_sequence(int begin, unsigned int end);

        /**
         * @param min Min range
         * @param max Max range
         * @param seed Seed random generator
         * @brief Fill tensor with random values followed by uniform distribution.
         */
        void fill_uniform(scalar_t min, scalar_t max, unsigned int seed);

        /**
         * @param mean Mean
         * @param stddev Standard deviation
         * @param seed Seed random generator
         * @brief Fill tensor with random values followed by normal distribution.
         */
        void fill_normal(scalar_t mean, scalar_t stddev, unsigned int seed);

        /**
         * @param seed Seed random generator
         * @brief Fill tensor with random values followed by uniform distribution and xavier limit.
         */
        void fill_xavier_uniform(unsigned int seed);

        /**
         * @param seed Seed random generator
         * @brief Fill tensor with random values followed by normal distribution and he standard deviation.
         */
        void fill_he_normal(unsigned int seed);

        /**
         * @brief Load the tensor from the binary file
         * @param filename Path of the binary file
         */
        void fill_from_binary(const std::string& filename);

        /**
         * @brief Save the tensor in the binary file
         * @param filename Path of the binary file
         */
        void save_to_binary(const std::string& filename) const;

        /**
         * @brief Print first/last rows/columns of tensor, with additional information.
         */
        void dump() const;

        /**
         * @brief Sum of tensor elements
         */
        static scalar_t sum(const Tensor& tensor);

        /**
         * @brief Mean of tensor elements
         */
        static scalar_t mean(const Tensor& tensor);

        /**
         * @brief Max of tensor elements
         */
        static scalar_t max(const Tensor& tensor);

        /**
         * @brief Min of tensor elements
         */
        static scalar_t min(const Tensor& tensor);

      private:
        // Number of rows
        unsigned int size_;
        // Number of columns
        unsigned int length_;
        std::vector<scalar_t> elements_;
    };
} // namespace stellar::core

#endif // ! TENSOR_H
