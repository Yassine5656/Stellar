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

#ifndef TENSOR_HH
#define TENSOR_HH

#include <vector>

#include "core/tensor/fwd.hh"
#include "misc/misc.hh"

namespace stellar::core::tensor
{
    /**
     * @brief Tensor class used for deep learning purposes.
     */
    class Tensor
    {
      public:
        struct Shape
        {
          public:
            Shape(const unsigned int rows, const unsigned int cols)
              : rows_{rows}
              , cols_{cols}
            {}
            unsigned int rows() const noexcept { return rows_; }
            unsigned int cols() const noexcept { return cols_; }

          private:
            unsigned int rows_;
            unsigned int cols_;
        };

      public:
        /**
         * @param rows Number of rows
         * @param cols Number of columns
         * @brief General tensor constructor
         */
        Tensor(const unsigned int rows_, const unsigned int cols_)
          : shape_{rows_, cols_}
          , size_{rows_ * cols_}
          , elements_(size_)
        {
            if (rows() == 0 || cols() == 0)
            {
                STELLAR_ERROR("invalid rows/cols number.\n");
            }
        }

        /**
         * @param size Number of elements
         * @param rows Number of rows
         * @param cols Number of columns
         * @attention Internal-Use Only constructor. It assumes: size = rows * cols
         */
        Tensor(const unsigned int size_,
               const unsigned int rows_,
               const unsigned int cols_)
          : shape_{rows_, cols_}
          , size_{size_}
          , elements_(size_)
        {
            if (rows() == 0 || cols() == 0)
            {
                STELLAR_ERROR("invalid rows/cols number.\n");
            }
        }

        /**
         * @param size Size of the squared tensor
         * @brief Square tensor constructor
         */
        explicit Tensor(const unsigned int size_)
          : shape_{size_, size_}
          , size_{size_ * size_}
          , elements_(this->size_)
        {
            if (size_ == 0)
            {
                STELLAR_ERROR("invalid size.\n");
            }
        }

        /**
         * @param other Matrix to copy
         * @brief Copy tensor constructor
         */
        Tensor(const Tensor& other) = default;
        /**
         * @param other Matrix to move
         * @brief Move tensor constructor
         */
        Tensor(Tensor&& other) noexcept = default;

        Tensor& operator=(const Tensor& other)     = default;
        Tensor& operator=(Tensor&& other) noexcept = default;
        ~Tensor()                                  = default;

        /**
         * @param other Matrix to copy
         * @param rows Number of sub-tensor's rows
         * @param cols Number of sub-tensor's columns
         * @param row_begin First row to copy
         * @param column_begin First column to copy
         * @brief Copy sub-tensor constructor
         */
        Tensor(const Tensor& other,
               unsigned int rows_,
               unsigned int cols_,
               unsigned int row_begin,
               unsigned int column_begin);

        /**
         * @brief Return the total number of Tensor's elements
         */
        [[nodiscard]] unsigned int size() const noexcept { return size_; }

        /**
         * @brief Return the number of Tensor's rows
         */
        [[nodiscard]] unsigned int rows() const noexcept { return shape_.rows(); }

        /**
         * @brief Return the number of Tensor's columns
         */
        [[nodiscard]] unsigned int cols() const noexcept { return shape_.cols(); }

        /**
         * @brief Return Tensor's shape
         */
        [[nodiscard]] const Shape& shape() const noexcept { return shape_; }

        /**
         * @brief INTERNAL USE ONLY!
         */
        [[nodiscard]] scalar_t* unsafe_data() noexcept { return elements_.data(); }

        /**
         * @brief INTERNAL USE ONLY!
         */
        [[nodiscard]] const scalar_t* unsafe_data() const noexcept
        {
            return elements_.data();
        }

        /**
         * @param row Row index
         * @param col Column index
         * @brief Get the element of the row-th row and col-th column.
         * Indexes are zero-based.
         */
        [[nodiscard]] scalar_t get(const unsigned int row, const unsigned int col) const
        {
            if (row >= rows() || col >= cols())
            {
                STELLAR_ERROR("invalid row/col.\n");
            }
            return elements_[row * cols() + col];
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
            if (row >= rows() || col >= cols())
            {
                STELLAR_ERROR("invalid row/col.\n");
            }

            elements_[row * cols() + col] = val;
        }

      private:
        // Tensor shape
        Shape shape_;
        // Number of elements
        unsigned int size_;
        // Data
        std::vector<scalar_t> elements_;
    };
} // namespace stellar::core::tensor

#endif // ! TENSOR_HH
