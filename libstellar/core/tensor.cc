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

#include "core/tensor.hh"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <omp.h>

using namespace stellar::core;

static constexpr std::string_view MAGIC_NUMBER{"stellar"};
constexpr unsigned int MAX_ELM_RENDERED   = 3;
constexpr unsigned int PADDING            = 10;
constexpr unsigned int PARALLEL_THRESHOLD = 3000;
constexpr unsigned int PRECISION_RENDERED = 4;

static inline void
WRITE_STELLAR_HEADER(std::ostream& os, const unsigned int size, const unsigned int length)
{
    os << MAGIC_NUMBER << '\n';
    os << size << '\n';
    os << length << '\n';
}

Tensor::Tensor(const Tensor& other,
               const unsigned int size,
               const unsigned int length,
               const unsigned int row_begin,
               const unsigned int column_begin)
  : size_{size}
  , length_{length}
  , elements_(size * length)
{
    if (size == 0 || length == 0)
    {
        STELLAR_ERROR("invalid row/col.\n");
    }
    // clang-format off
    #pragma omp parallel for default(none) shared(other, size, length, row_begin, column_begin) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int i = 0; i < size; ++i)
        for (unsigned int j = 0; j < length; j++)
            elements_[i * length + j] = other.get(i + row_begin, j + column_begin);
}

void Tensor::add(const Tensor& other)
{
    // clang-format off
    #pragma omp parallel for default(none) shared(other) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int index = 0; index < getSize() * getLength(); ++index)
        elements_[index] += other.elements_[index];
}

void Tensor::sub(const Tensor& other)
{
    // clang-format off
    #pragma omp parallel for default(none) shared(other) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int index = 0; index < getSize() * getLength(); ++index)
        elements_[index] -= other.elements_[index];
}

void Tensor::multiply(const Tensor& other)
{
    if (getLength() != other.getSize())
    {
        STELLAR_ERROR("multiplication forbidden\n");
    }
    // In-place tensor result
    Tensor result{getSize(), other.getLength()};

    // clang-format off
    #pragma omp parallel for default(none) shared(other, result) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int row = 0; row < getSize(); ++row)
    {
        for (unsigned int index = 0; index < getLength(); ++index)
        {
            const float temp = get(row, index);
            for (unsigned int column = 0; column < other.getLength(); column++)
                result.elements_[row * result.getLength() + column] +=
                  temp * other.get(index, column);
        }
    }

    *this = std::move(result);
}

void Tensor::multiplyScalar(scalar_t scalar)
{
    // clang-format off
    #pragma omp parallel for default(none) reduction(* : scalar) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int index = 0; index < getSize() * getLength(); ++index)
        elements_[index] *= scalar;
}

void Tensor::fill_constant(const scalar_t scalar)
{
    // clang-format off
    #pragma omp parallel for default(none) shared(scalar) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int i = 0; i < getSize() * getLength(); ++i)
        elements_[i] = scalar;
}

void Tensor::fill_diagonal(const scalar_t scalar)
{
    // Assert square tensor
    if (getSize() != getLength())
    {
        STELLAR_ERROR("Tensor is not square one.\n");
    }
    // clang-format off
    #pragma omp parallel for default(none) shared(scalar) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int row = 0; row < getSize(); ++row)
    {
        for (unsigned col = 0; col < row; ++col)
            elements_[row * getLength() + col] = 0;
        for (unsigned col = row + 1; col < getLength(); ++col)
            elements_[row * getLength() + col] = 0;
        elements_[row * getLength() + row] = scalar;
    }
}

void Tensor::fill_triangular(const scalar_t scalar, const bool upper_section)
{
    // Assert square matrix
    if (getSize() != getLength())
    {
        STELLAR_ERROR("Tensor is not square one.\n");
    }
    unsigned int end_row = getSize();
    if (upper_section)
    {
        ++end_row;
    }
    // clang-format off
    #pragma omp parallel default(none) shared(scalar, upper_section, end_row) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    {
        // clang-format off
        #pragma omp for
        // clang-format on
        for (unsigned int row = 0; row < getSize(); ++row)
        {
            for (unsigned int col = 0; col < row; ++col)
            {
                elements_[row * getLength() + col] = 0;
            }
            unsigned int col = row;
            if (!upper_section)
            {
                ++col;
            }
            for (; col < getSize(); ++col)
            {
                elements_[row * getLength() + col] = scalar;
            }
        }
    }
}

void Tensor::fill_sequence(const int begin, const unsigned int end)
{
    // Assert increasing sequence
    if (begin > static_cast<int>(end))
    {
        STELLAR_ERROR("Tensor is not square one.\n");
    }
    // clang-format off
    #pragma omp parallel for default(none) shared(begin, end) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int index = 0; index < getSize() * getLength(); ++index)
    {
        elements_[index] =
          static_cast<scalar_t>(begin + static_cast<int>(index % (end + 1)));
    }
}

void Tensor::fill_uniform(const scalar_t min, const scalar_t max, const unsigned int seed)
{
    // clang-format off
    #pragma omp parallel default(none) shared(min, max, seed) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    {
        const int thread_id = omp_get_thread_num();
        auto generator      = std::mt19937(seed + thread_id);
        auto distribution   = std::uniform_real_distribution(min, max);
        // clang-format off
        #pragma omp for
        // clang-format on
        for (unsigned int index = 0; index < getSize() * getLength(); ++index)
        {
            elements_[index] = distribution(generator);
        }
    }
}

void Tensor::fill_normal(const scalar_t mean,
                         const scalar_t stddev,
                         const unsigned int seed)
{
    // clang-format off
    #pragma omp parallel default(none) shared(mean, stddev, seed) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    {
        const int thread_id = omp_get_thread_num();
        auto generator      = std::mt19937(seed + thread_id);
        auto distribution   = std::normal_distribution(mean, stddev);
        // clang-format off
        #pragma omp for
        // clang-format on
        for (unsigned int index = 0; index < getSize() * getLength(); ++index)
        {
            elements_[index] = distribution(generator);
        }
    }
}

void Tensor::fill_xavier_uniform(const unsigned int seed)
{
    scalar_t xavier_scalar =
      std::sqrt(6.0f / static_cast<scalar_t>(getSize() + getLength()));
    // clang-format off
    #pragma omp parallel default(none) shared(xavier_scalar, seed) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    {
        const int thread_id = omp_get_thread_num();
        auto generator      = std::mt19937(seed + thread_id);
        auto distribution = std::uniform_real_distribution(-xavier_scalar, xavier_scalar);
        // clang-format off
        #pragma omp for
        // clang-format on
        for (unsigned int index = 0; index < getSize() * getLength(); ++index)
        {
            elements_[index] = distribution(generator);
        }
    }
}

void Tensor::fill_he_normal(const unsigned int seed)
{
    scalar_t he_scalar = std::sqrt(2.0f / static_cast<scalar_t>(getLength()));
    // clang-format off
    #pragma omp parallel default(none) shared(he_scalar, seed) if (getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    {
        const int thread_id = omp_get_thread_num();
        auto generator      = std::mt19937(seed + thread_id);
        auto distribution   = std::normal_distribution<scalar_t>(0, he_scalar);
        // clang-format off
        #pragma omp for
        // clang-format on
        for (unsigned int index = 0; index < getSize() * getLength(); ++index)
        {
            elements_[index] = distribution(generator);
        }
    }
}

bool check_magic_number(std::ifstream& os)
{
    std::string line;
    std::getline(os, line);

    return line == MAGIC_NUMBER;
}

std::array<char, 4> scalar_t_to_bytes(const scalar_t scalar)
{
    std::array<char, 4> res;
    std::memcpy(res.data(), &scalar, sizeof(scalar_t));
    return res;
}

void Tensor::save_to_binary(const std::string& filename) const
{
    std::ofstream file{filename, std::ios::binary};
    if (!file.is_open())
    {
        STELLAR_ERROR("File doesn't exist. filename: '" + filename + "'\n");
        return;
    }
    // Write stellar-format header
    WRITE_STELLAR_HEADER(file, getSize(), getLength());

    for (unsigned int row = 0; row < getSize(); ++row)
    {
        for (unsigned int column = 0; column < getLength(); ++column)
        {
            std::array<char, 4> binary_weight;
            binary_weight = scalar_t_to_bytes(get(row, column));
            for (unsigned int i = 0; i < 4; ++i)
            {
                file << binary_weight[i];
            }
        }
    }

    file.close();
}

void Tensor::fill_from_binary(const std::string& filename)
{
    std::ifstream file{filename, std::ios::binary};
    if (!file.is_open())
    {
        STELLAR_ERROR("File does not exist! Filename: '" + filename + "'\n");
        return;
    }
    if (!check_magic_number(file))
    {
        STELLAR_ERROR("Invalid Magic Number!\n");
        file.close();
        return;
    }
    std::string line;
    std::getline(file, line);
    const unsigned int rows = std::strtoul(line.data(), nullptr, 10);
    std::getline(file, line);
    const unsigned int cols = std::strtoul(line.data(), nullptr, 10);
    // TODO: Create Tensor
    //Tensor resulted_tensor{rows, cols};

    for (unsigned int row = 0; row < rows; ++row)
    {
        for (unsigned int col = 0; col < cols; ++col)
        {
            std::array<char, 4> arr;
            if (!file.read(arr.data(), sizeof(scalar_t)))
            {
                STELLAR_ERROR("Premature End of File at row " + std::to_string(row));
                file.close();
                return;
            }

            scalar_t scalar;
            std::memcpy(&scalar, arr.data(), sizeof(scalar_t));
            set(row, col, scalar);
        }
    }
    if (std::string dummy; file.read(dummy.data(), 1))
    {
        STELLAR_ERROR("File is larger than expected (trailing characters)!\n");
        file.close();
        return;
    }

    file.close();
}

void Tensor::dump() const
{
    const auto backup_flags = std::cout.flags();
    std::cout << std::setprecision(PRECISION_RENDERED);
    // Dump dimensions
    std::cout << "Tensor: [" << getSize() << " x " << getLength() << "]" << "\n";
    const unsigned int rows_rendered = std::min(MAX_ELM_RENDERED, getSize());
    const unsigned int cols_rendered = std::min(MAX_ELM_RENDERED, getLength());

    // Dump tensor
    std::cout << "[" << "\n";
    for (unsigned int row = 0; row < rows_rendered; ++row)
    {
        std::cout << std::setw(PADDING) << " " << elements_[row * getLength()];
        unsigned int col = 1;
        for (; col < cols_rendered; ++col)
        {
            std::cout << ", " << std::setw(PADDING) << elements_[row * getLength() + col];
        }
        if (getLength() - cols_rendered > col)
        {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        for (col = std::max(col, getLength() - cols_rendered); col < getLength(); ++col)
        {
            std::cout << ", " << std::setw(PADDING) << elements_[row * getLength() + col];
        }
        std::cout << "\n";
    }

    if (getSize() - rows_rendered > rows_rendered)
    {
        std::cout << " " << std::setw(PADDING) << "...";
        unsigned int col = 1;
        for (; col < cols_rendered; ++col)
        {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        if (getLength() - cols_rendered > col)
        {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        for (col = std::max(col, getLength() - cols_rendered); col < getLength(); ++col)
        {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        std::cout << "\n";
    }

    for (unsigned int row = std::max(rows_rendered, getSize() - rows_rendered);
         row < getSize(); ++row)
    {
        std::cout << std::setw(PADDING) << " " << elements_[row * getLength()];
        unsigned int col = 1;
        for (; col < cols_rendered; ++col)
        {
            std::cout << ", " << std::setw(PADDING) << elements_[row * getLength() + col];
        }
        if (getLength() - cols_rendered > col)
        {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        for (col = std::max(col, getLength() - cols_rendered); col < getLength(); ++col)
        {
            std::cout << ", " << std::setw(PADDING) << elements_[row * getLength() + col];
        }
        std::cout << "\n";
    }
    std::cout << "]" << "\n";

    // Dump tensor's characteristics
    std::cout << "Min: " << min(*this) << "\n";
    std::cout << "Max: " << max(*this) << "\n";
    std::cout << "Mean: " << mean(*this) << "\n";

    // Restore cout flags
    std::cout.flags(backup_flags);
}

scalar_t Tensor::sum(const Tensor& tensor)
{
    scalar_t sum = 0.0f;
    // clang-format off
    #pragma omp parallel for default(none) shared(tensor) reduction(+ : sum) if (tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int index = 0; index < tensor.getSize() * tensor.getLength(); ++index)
        sum += tensor.elements_[index];

    return sum;
}

scalar_t Tensor::mean(const Tensor& tensor)
{
    scalar_t sum = 0.0f;
    // clang-format off
    #pragma omp parallel for default(none) shared(tensor) reduction(+ : sum) if (tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    for (unsigned int index = 0; index < tensor.getSize() * tensor.getLength(); ++index)
        sum += tensor.elements_[index];

    return sum / static_cast<scalar_t>(tensor.getSize() * tensor.getLength());
}

scalar_t Tensor::max(const Tensor& tensor)
{
    scalar_t max = std::numeric_limits<scalar_t>::lowest();

    // clang-format off
    #pragma omp parallel default(none) shared(tensor, max) if (tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    {
        scalar_t local_max = std::numeric_limits<scalar_t>::lowest();

        // clang-format off
        #pragma omp for nowait
        // clang-format on
        for (unsigned int index = 0; index < tensor.getSize() * tensor.getLength();
             ++index)
            if (local_max < tensor.elements_[index])
                local_max = tensor.elements_[index];

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

scalar_t Tensor::min(const Tensor& tensor)
{
    scalar_t min = std::numeric_limits<scalar_t>::max();

    // clang-format off
    #pragma omp parallel default(none) shared(tensor, min) if (tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    // clang-format on
    {
        scalar_t local_min = std::numeric_limits<scalar_t>::max();

        // clang-format off
        #pragma omp for nowait
        // clang-format on
        for (unsigned int index = 0; index < tensor.getSize() * tensor.getLength();
             ++index)
            if (tensor.elements_[index] < local_min)
                local_min = tensor.elements_[index];

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