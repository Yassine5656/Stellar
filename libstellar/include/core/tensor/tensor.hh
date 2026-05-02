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
        /**
         * @param rows Number of rows
         * @param cols Number of columns
         * @brief General tensor constructor
         */
        Tensor(const unsigned int rows, const unsigned int cols)
          : shape_{rows, cols}
          , size_{rows * cols}
          , elements_(size_)
        {
            if (getRows() == 0 || getCols() == 0)
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
        Tensor(const unsigned int size, const unsigned int rows, const unsigned int cols)
          : shape_{rows, cols}
          , size_{size}
          , elements_(size)
        {
            if (getRows() == 0 || getCols() == 0)
            {
                STELLAR_ERROR("invalid rows/cols number.\n");
            }
        }

        /**
         * @param size Size of the squared tensor
         * @brief Square tensor constructor
         */
        explicit Tensor(const unsigned int size)
          : shape_{size, size}
          , size_{size * size}
          , elements_(size * size)
        {
            if (size == 0)
            {
                STELLAR_ERROR("invalid size.\n");
            }
        }

        /**
         * @param other Matrix to move
         * @brief Move tensor constructor
         */
        Tensor(const Tensor& other) noexcept = default;

        Tensor& operator=(Tensor&&) noexcept = delete;

        /**
         * @param other Matrix to copy
         * @param rows Number of sub-tensor's rows
         * @param cols Number of sub-tensor's columns
         * @param row_begin First row to copy
         * @param column_begin First column to copy
         * @brief Copy sub-tensor constructor
         */
        Tensor(const Tensor& other,
               unsigned int rows,
               unsigned int cols,
               unsigned int row_begin,
               unsigned int column_begin);

        /**
         * @brief Return the total number of Tensor's elements
         */
        [[nodiscard]] unsigned int getSize() const { return size_; }

        /**
         * @brief Return the number of tensor's rows
         */
        [[nodiscard]] unsigned int getRows() const { return shape_.rows; }

        /**
         * @brief Return the number of tensor's columns
         */
        [[nodiscard]] unsigned int getCols() const { return shape_.cols; }

        /**
         * @brief INTERNAL USE ONLY!
         */
        [[nodiscard]] scalar_t* unsafe_data() { return elements_.data(); }

        /**
         * @brief INTERNAL USE ONLY!
         */
        [[nodiscard]] const scalar_t* unsafe_data() const { return elements_.data(); }

        /**
         * @param row Row index
         * @param col Column index
         * @brief Get the element of the row-th row and col-th column.
         * Indexes are zero-based.
         */
        [[nodiscard]] scalar_t get(const unsigned int row, const unsigned int col) const
        {
            if (row >= getRows() || col >= getCols())
            {
                STELLAR_ERROR("invalid row/col.\n");
            }
            return elements_[row * getCols() + col];
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
            if (row >= getRows() || col >= getCols())
            {
                STELLAR_ERROR("invalid row/col.\n");
            }

            elements_[row * getCols() + col] = val;
        }

      private:
        // Tensor shape
        struct shape
        {
            const unsigned int rows;
            const unsigned int cols;
            shape(const unsigned int rows_, const unsigned int cols_)
              : rows{rows_}
              , cols{cols_} {};
        } shape_;
        // Number of elements
        const unsigned int size_;
        // Data
        std::vector<scalar_t> elements_;
    };
} // namespace stellar::core::tensor

#endif // ! TENSOR_HH
