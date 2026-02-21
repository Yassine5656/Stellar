//
// Created by yassine on 2/12/26.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <cassert>
#include <omp.h>
#include <random>
#include <vector>

namespace stellar {
    using scalar_t = float;

    /**
     * @brief Tensor class used for deep learning purposes.
     */
    class Tensor {
    public:
        /**
         * @param size Number of rows
         * @param length Number of columns
         * @brief General tensor constructor
         */
        Tensor(const unsigned int size, const unsigned int length) : Tensor(
            size, length, 0) {
        }

        /**
         * @param size Dimension
         * @brief Square tensor constructor
         */
        explicit Tensor(const unsigned int size) : Tensor(size, size, 0) {
        }

        /**
         * @param other Matrix to copy
         * @brief Copy tensor constructor
         */
        Tensor(const Tensor &other) : size_(other.getSize()),
                                      length_(other.getLength()) {
            assert(size_ != 0 && length_ != 0);
            for (unsigned int i = 0; i < size_; i++)
                for (unsigned int j = 0; j < length_; j++)
                    elements_.push_back(other.get(i, j));
        }

        /**
         * @param other Matrix to copy
         * @param size Number of sub-tensor's rows
         * @param length Number of sub-tensor's columns
         * @param row_begin First row to copy
         * @param column_begin First column to copy
         * @brief Copy sub-tensor constructor
         */
        Tensor(const Tensor &other, const unsigned int size,
               const unsigned int length, const unsigned int row_begin,
               const unsigned int column_begin) : size_(size), length_(length),
                                                  elements_(
                                                      std::vector<scalar_t>(
                                                          size * length)) {
            assert(size_ != 0 && length_ != 0);
            for (unsigned int i = 0; i < size; i++)
                for (unsigned int j = 0; j < length; j++)
                    elements_[i * length + j] = other.get(
                        i + row_begin, j + column_begin);
        }

        [[nodiscard]] unsigned int getSize() const {
            return size_;
        }

        [[nodiscard]] unsigned int getLength() const {
            return length_;
        }

        /**
         * @param row Row index
         * @param col Column index
         * @brief Get the element of the row-th row and col-th column.
         * Indexes are zero-based.
         */
        [[nodiscard]] scalar_t get(const unsigned int row,
                                   const unsigned int col) const {
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
        void set(const unsigned int row, const unsigned int col,
                 const scalar_t val) {
            assert(row < size_ && col < length_);
            elements_[row * length_ + col] = val;
        }

        /**
         * @brief Tensor addition by parallelism of calculations and cache optimization.
         */
        void add(const Tensor &other);

        /**
         * @brief Tensor substraction by parallelism of calculations and cache optimization.
         */
        void sub(const Tensor &other);

        /**
         * @brief Tensor product by parallelism of calculations and cache optimization.
         */
        void multiply(const Tensor &other) {
            multiply(other, 0, size_);
        }

        /**
         * @param scalar Scalar coefficient
         * @brief Tensor scalar product
         */
        void multiplyScalar(int scalar);

        /**
         * @param scalar New value
         * @brief Set all tensor elements to 'scalar'.
         */
        void fill_constant(const scalar_t scalar) {
            for (unsigned int i = 0; i < size_ * length_; i++)
                elements_[i] = scalar;
        }

        /**
         * @param scalar New value
         * @brief Fill the main diagonal by 'scalar'.
         */
        void fill_diagonal(const scalar_t scalar) {
            // Assert square tensor
            assert(size_ == length_);
            for (unsigned int row = 0; row < size_; row++) {
                for (unsigned col = 0; col < row; col++)
                    elements_[row * length_ + col] = 0;
                for (unsigned col = row + 1; col < length_; col++)
                    elements_[row * length_ + col] = 0;
                elements_[row * length_ + row] = scalar;
            }
        }

        /**
         * @brief Transform tensor to identity.
         * Thus, the main diagonal is set to 1, and other elements to 0.
         */
        void fill_identity() {
            fill_diagonal(1);
        }

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
        void fill_sequence(const unsigned int begin, const unsigned int end) {
            // Assert increasing sequence
            assert(begin <= end);
            for (unsigned int index = 0; index < size_ * length_; index++)
                elements_[index] = static_cast<scalar_t>(begin + index % end);
        }

        /**
         * @param min Min range
         * @param max Max range
         * @param seed Seed random generator
         * @brief Fill tensor with random values followed by uniform distribution.
         */
        void fill_uniform(const scalar_t min, const scalar_t max,
                          const unsigned int seed) {
            auto generator = std::mt19937(seed);
            auto distribution = std::uniform_real_distribution(min, max);
            for (unsigned int index = 0; index < size_ * length_; index++) {
                elements_[index] = distribution(generator);
            }
        }

        /**
         * @param mean Mean
         * @param stddev Standard deviation
         * @param seed Seed random generator
         * @brief Fill tensor with random values followed by normal distribution.
         */
        void fill_normal(const scalar_t mean, const scalar_t stddev,
                         const unsigned int seed) {
            auto generator = std::mt19937(seed);
            auto distribution = std::normal_distribution(mean, stddev);
            for (unsigned int index = 0; index < size_ * length_; index++) {
                elements_[index] = distribution(generator);
            }
        }

        /**
         * @param seed Seed random generator
         * @brief Fill tensor with random values followed by uniform distribution and xavier limit.
         */
        void fill_xavier_uniform(const unsigned int seed) {
            const scalar_t xavier_scalar = std::sqrtf(
                6.0f / static_cast<scalar_t>(size_ + length_));
            #pragma omp parallel default(none) shared(xavier_scalar, seed)
            {
                const int thread_id = omp_get_thread_num();
                auto generator = std::mt19937(seed + thread_id);
                auto distribution = std::uniform_real_distribution(
                    -xavier_scalar, xavier_scalar);
                #pragma omp for
                for (unsigned int index = 0; index < size_ * length_; index++) {
                    elements_[index] = distribution(generator);
                }
            }
        }

        /**
         * @brief Load the tensor from the binary file
         * @param filename Path of the binary file
         */
        void fill_from_binary(std::string filename);

        /**
         * @brief Print first/last rows/columns of tensor, with additional information.
         */
        void dump() const;

        /**
         * @brief Sum of tensor elements
         */
        static scalar_t sum(const Tensor &tensor);

        /**
         * @brief Mean of tensor elements
         */
        static scalar_t mean(const Tensor &tensor);

        /**
         * @brief Max of tensor elements
         */
        static scalar_t max(const Tensor &tensor);

        /**
         * @brief Min of tensor elements
         */
        static scalar_t min(const Tensor &tensor);

        /**
         * @brief Check if NaN values are present in the tensor.
         * @param tensor Tensor
         */
        static bool check_invalid(const Tensor &tensor);

    private:
        unsigned int size_;
        unsigned int length_;
        std::vector<scalar_t> elements_;

        /**
         * @param size Number of rows
         * @param length Number of columns
         * @param c Value to set
         * @brief Private constructor, with default value elements initialization.
         */
        Tensor(const unsigned int size, const unsigned int length,
               const scalar_t c)
            : size_(size),
              length_(length), elements_(std::vector(size * length, c)) {
            assert(size_ != 0 && length_ != 0);
        }

        void multiply(Tensor other, unsigned int begin_row,
                      unsigned int end_row);
    };
}

#endif // ! TENSOR_H
