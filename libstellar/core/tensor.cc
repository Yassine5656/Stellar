//
// Created by yassine on 2/12/26.
//

#include "../../include/core/tensor.hh"

#include <bitset>
#include <fstream>
#include <iomanip>
#include <iostream>

#define MAX_ELM_RENDERED ((unsigned int) 3)
#define PADDING 10
#define PRECISION_RENDERED 4

constexpr size_t PARALLEL_THRESHOLD = 3000;

namespace {
    template <typename Type>
    concept double_weights = requires(Type x)
    {
        { x } -> std::same_as<double>;
    };
}

stellar::core::Tensor::Tensor(const Tensor &other, const unsigned int size,
                        const unsigned int length, const unsigned int row_begin,
                        const unsigned int column_begin) : size_(size),
    length_(length),
    elements_(size * length) {
    assert(size_ != 0 && length_ != 0);
    #pragma omp parallel for default(none) \
        shared(other, size, length, row_begin, column_begin) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int i = 0; i < size; i++)
        for (unsigned int j = 0; j < length; j++)
            elements_[i * length + j] = other.get(
                i + row_begin, j + column_begin);
}

void stellar::core::Tensor::add(const Tensor &other) {
    #pragma omp parallel for default(none) shared(other) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int index = 0; index < size_ * length_; index++)
        elements_[index] += other.elements_[index];
}

void stellar::core::Tensor::sub(const Tensor &other) {
    #pragma omp parallel for default(none) shared(other) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int index = 0; index < size_ * length_; index++)
        elements_[index] -= other.elements_[index];
}

void stellar::core::Tensor::multiply(const Tensor &other) {
    assert(length_ == other.getSize());
    // In-place tensor result
    auto result = Tensor(size_, other.getLength());

    #pragma omp parallel for default(none) shared(other, result) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int row = 0; row < size_; row++) {
        for (unsigned int index = 0; index < length_; index++) {
            const float temp = get(row, index);
            for (unsigned int column = 0; column < other.getLength(); column++)
                result.elements_[row * result.getLength() + column] += temp *
                        other.get(index, column);
        }
    }

    *this = std::move(result);
}

void stellar::core::Tensor::multiplyScalar(scalar_t scalar) {
    #pragma omp parallel for default(none) reduction(*: scalar) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int index = 0; index < size_ * length_; index++)
        elements_[index] *= scalar;
}

void stellar::core::Tensor::fill_constant(const scalar_t scalar) {
    #pragma omp parallel for default(none) shared(scalar) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int i = 0; i < size_ * length_; i++)
        elements_[i] = scalar;
}

void stellar::core::Tensor::fill_diagonal(const scalar_t scalar) {
    // Assert square tensor
    assert(size_ == length_);
    #pragma omp parallel for default(none) shared(scalar) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int row = 0; row < size_; row++) {
        for (unsigned col = 0; col < row; col++)
            elements_[row * length_ + col] = 0;
        for (unsigned col = row + 1; col < length_; col++)
            elements_[row * length_ + col] = 0;
        elements_[row * length_ + row] = scalar;
    }
}

void stellar::core::Tensor::fill_triangular(const scalar_t scalar,
                                      const bool upper_section) {
    // Assert square matrix
    assert(size_ == length_);
    #pragma omp parallel default(none) shared(scalar, upper_section) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    {
        if (upper_section) {
            #pragma omp for
            for (unsigned int row = 0; row < size_; row++) {
                for (unsigned int col = 0; col < row; col++)
                    elements_[row * length_ + col] = 0;
                for (unsigned int col = row; col <= size_; col++)
                    elements_[row * length_ + col] = scalar;
            }
        } else {
            #pragma omp for
            for (unsigned int row = 0; row < size_; row++) {
                for (unsigned int col = 0; col <= row; col++)
                    elements_[row * length_ + col] = scalar;
                for (unsigned int col = row + 1; col < size_; col++)
                    elements_[row * length_ + col] = 0;
            }
        }
    }
}

void stellar::core::Tensor::fill_sequence(const int begin,
                                    const unsigned int end) {
    // Assert increasing sequence
    assert(begin <= static_cast<int>(end));
    #pragma omp parallel for default(none) shared(begin, end) \
        if(getLength() >= PARALLEL_THRESHOLD || getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int index = 0; index < size_ * length_; index++)
        elements_[index] = static_cast<scalar_t>(begin + static_cast<int>(index % (end + 1)));
}

void stellar::core::Tensor::fill_uniform(const scalar_t min, const scalar_t max,
                                   const unsigned int seed) {
    #pragma omp parallel default(none) shared(min, max, seed) \
        if(length_ >= PARALLEL_THRESHOLD || size_ >= PARALLEL_THRESHOLD)
    {
        const int thread_id = omp_get_thread_num();
        auto generator = std::mt19937(seed + thread_id);
        auto distribution = std::uniform_real_distribution(min, max);
        #pragma omp for
        for (unsigned int index = 0; index < size_ * length_; index++) {
            elements_[index] = distribution(generator);
        }
    }
}

void stellar::core::Tensor::fill_normal(const scalar_t mean, const scalar_t stddev,
                                  const unsigned int seed) {
    #pragma omp parallel default(none) shared(mean, stddev, seed) \
    if(length_ >= PARALLEL_THRESHOLD || size_ >= PARALLEL_THRESHOLD)
    {
        const int thread_id = omp_get_thread_num();
        auto generator = std::mt19937(seed + thread_id);
        auto distribution = std::normal_distribution(mean, stddev);
        #pragma omp for
        for (unsigned int index = 0; index < size_ * length_; index++) {
            elements_[index] = distribution(generator);
        }
    }
}

void stellar::core::Tensor::fill_xavier_uniform(const unsigned int seed) {
    scalar_t xavier_scalar;
    if (typeid(scalar_t) == typeid(float)) {
        xavier_scalar = std::sqrtf(6.0f / static_cast<scalar_t>(size_ + length_));
    }
    else {
        xavier_scalar = std::sqrt(6.0f / static_cast<scalar_t>(size_ + length_));
    }
    #pragma omp parallel default(none) shared(xavier_scalar, seed) \
        if(length_ >= PARALLEL_THRESHOLD || size_ >= PARALLEL_THRESHOLD)
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

void stellar::core::Tensor::fill_he_normal(const unsigned int seed) {
    scalar_t he_scalar;
    if (typeid(scalar_t) == typeid(float)) {
        he_scalar = std::sqrtf(2.0f / static_cast<scalar_t>(length_));
    }
    else
    {
        he_scalar = std::sqrt(2.0f / static_cast<scalar_t>(length_));
    }
    #pragma omp parallel default(none) shared(he_scalar, seed) \
        if(length_ >= PARALLEL_THRESHOLD || size_ >= PARALLEL_THRESHOLD)
    {
        const int thread_id = omp_get_thread_num();
        auto generator = std::mt19937(seed + thread_id);
        auto distribution = std::normal_distribution<scalar_t>(0, he_scalar);
        #pragma omp for
        for (unsigned int index = 0; index < size_ * length_; index++) {
            elements_[index] = distribution(generator);
        }
    }
}

template <typename scalar_type = stellar::core::scalar_t>
scalar_type compute_binary_to_scalar(const std::string& bin) requires double_weights<scalar_type>
{
    return std::bit_cast<scalar_type>(std::stol(bin, nullptr, 2));
}

template <typename scalar_type = stellar::core::scalar_t>
scalar_type compute_binary_to_scalar(const std::string& bin)
{
    return std::bit_cast<scalar_type>(std::stoi(bin, nullptr, 2));
}

void stellar::core::Tensor::fill_from_binary(const std::string& filename) {
    assert(!filename.empty());
    std::ifstream file{filename};
    assert(file.is_open());
    if (!file.is_open()) {
        std::cerr << "fill_from_binary: File does not exist! Filepath: '" << filename << "'\n";
        return;
    }

    // FIXME: Check if it's a valid digit

    std::string line;
    unsigned int col = 0;
    if (!getline(file, line).eof()) {
        std::string word;
        std::stringstream token{line};
        while (token >> word) {
            const auto scalar = compute_binary_to_scalar(word);
            set(0, col, scalar);
            ++col;
        }
    }

    unsigned int row = 1;
    while (!getline(file, line).eof()) {
        unsigned int curr_col = 0;
        if (line.empty()) {
            continue;
        }
        std::string word;
        std::stringstream token{line};
        while (token >> word) {
            //const auto scalar = std::bit_cast<scalar_t>(std::stoi(word, nullptr, 2));
            const auto scalar = 0;
            set(row, curr_col, scalar);
            ++curr_col;
        }
        assert(curr_col == col);
        if (curr_col != col) {
            std::cerr << "fill_from_binary: File format invalid! Filepath: '" << filename << "'\n";
            file.close();
            return;
        }
        ++row;
    }

    file.close();
}

template <typename bit_type, typename weight_type, typename scalar_type>
std::string type_to_binary_string(const scalar_type scalar) {
    const auto bits = std::bit_cast<bit_type>(static_cast<weight_type>(scalar));
    return std::bitset<sizeof(scalar_type) * 8>(bits).to_string();
}

template<typename scalar_type = stellar::core::scalar_t>
std::string scalar_type_to_binary_string(const scalar_type scalar) requires double_weights<scalar_type>
{
    return type_to_binary_string<std::int64_t, double, scalar_type>(scalar);
}

template<typename scalar_type = stellar::core::scalar_t>
std::string scalar_type_to_binary_string(const scalar_type scalar)
{
    return type_to_binary_string<std::int32_t, float, scalar_type>(scalar);
}

void stellar::core::Tensor::save_to_binary(const std::string& filepath) const {
    assert(!filepath.empty());
    std::ofstream file{filepath};
    assert(file.is_open());
    if (!file.is_open()) {
        std::cerr << "File doesn't exist. Filepath : '" << filepath << "'\n";
        return;
    }

    for (unsigned int row = 0; row < size_; ++row) {
        std::string binary_weight = scalar_type_to_binary_string<>(get(row, 0));
        file << binary_weight;
        for (unsigned int column = 1; column < length_; ++column) {
            binary_weight = scalar_type_to_binary_string<>(get(row, column));
            file << " " << binary_weight;
        }
        file << "\n";
    }

    file.close();
}

void stellar::core::Tensor::dump() const {
    const auto backup_flags = std::cout.flags();
    std::cout << std::setprecision(PRECISION_RENDERED);
    std::cout << "Tensor: [" << size_ << " x " << length_ << "]" << "\n";
    const unsigned int rows_rendered = std::min(MAX_ELM_RENDERED, size_);
    const unsigned int cols_rendered = std::min(MAX_ELM_RENDERED, length_);
    std::cout << "[" << "\n";
    for (unsigned int row = 0; row < rows_rendered; row++) {
        std::cout << std::setw(PADDING) << " " << elements_[row * length_];
        unsigned int col = 1;
        for (; col < cols_rendered; col++) {
            std::cout << ", " << std::setw(PADDING) << elements_[
                row * length_ + col];
        }
        if (length_ - cols_rendered > col) {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        for (col = std::max(col, length_ - cols_rendered); col < length_; col
             ++) {
            std::cout << ", " << std::setw(PADDING) << elements_[
                row * length_ + col];
        }
        std::cout << "\n";
    }

    if (size_ - rows_rendered > rows_rendered) {
        std::cout << " " << std::setw(PADDING) << "...";
        unsigned int col = 1;
        for (; col < cols_rendered; col++) {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        if (length_ - cols_rendered > col) {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        for (col = std::max(col, length_ - cols_rendered); col < length_; col
             ++) {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        std::cout << "\n";
    }

    for (unsigned int row = std::max(rows_rendered, size_ - rows_rendered);
         row < size_; row++) {
        std::cout << std::setw(PADDING) << " " << elements_[row * length_];
        unsigned int col = 1;
        for (; col < cols_rendered; col++) {
            std::cout << ", " << std::setw(PADDING) << elements_[
                row * length_ + col];
        }
        if (length_ - cols_rendered > col) {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        for (col = std::max(col, length_ - cols_rendered); col < length_; col
             ++) {
            std::cout << ", " << std::setw(PADDING) << elements_[
                row * length_ + col];
        }
        std::cout << "\n";
    }
    std::cout << "]" << "\n";

    std::cout << "Min: " << min(*this) << "\n";
    std::cout << "Max: " << max(*this) << "\n";
    std::cout << "Mean: " << mean(*this) << "\n";
    std::cout << "NaN Present: " << std::boolalpha << check_NaN_values(*this) << "\n";

    std::cout.flags(backup_flags);
}

stellar::core::scalar_t stellar::core::Tensor::sum(const Tensor &tensor) {
    scalar_t sum = 0.0f;
    #pragma omp parallel for default(none) shared(tensor) reduction(+: sum) \
        if(tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int index = 0; index < tensor.size_ * tensor.length_; index++)
        sum += tensor.elements_[index];

    return sum;
}

stellar::core::scalar_t stellar::core::Tensor::mean(const Tensor &tensor) {
    scalar_t sum = 0.0f;
    #pragma omp parallel for default(none) shared(tensor) reduction(+: sum) \
        if(tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int index = 0; index < tensor.size_ * tensor.length_; index++)
        sum += tensor.elements_[index];

    return sum / static_cast<scalar_t>(tensor.getSize() * tensor.getLength());
}

stellar::core::scalar_t stellar::core::Tensor::max(const Tensor &tensor) {
    scalar_t max = std::numeric_limits<scalar_t>::lowest();

    #pragma omp parallel default(none) shared(tensor, max) \
        if(tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    {
        scalar_t local_max = std::numeric_limits<scalar_t>::lowest();

        #pragma omp for nowait
        for (unsigned int index = 0; index < tensor.size_ * tensor.length_; index++)
            if (local_max < tensor.elements_[index])
                local_max = tensor.elements_[index];

        #pragma omp critical
        {
            if (local_max > max) {
                max = local_max;
            }
        }
    }

    return max;
}

stellar::core::scalar_t stellar::core::Tensor::min(const Tensor &tensor) {
    scalar_t min = std::numeric_limits<scalar_t>::max();

    #pragma omp parallel default(none) shared(tensor, min) \
    if(tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    {
        scalar_t local_min = std::numeric_limits<scalar_t>::max();

        #pragma omp for nowait
        for (unsigned int index = 0; index < tensor.size_ * tensor.length_; index++)
            if (tensor.elements_[index] < local_min)
                local_min = tensor.elements_[index];

        #pragma omp critical
        {
            if (local_min < min) {
                min = local_min;
            }
        }
    }

    return min;
}

bool stellar::core::Tensor::check_NaN_values(const Tensor &tensor) {
    bool is_NaN_present = false;

    #pragma omp parallel for default(none) shared(tensor, is_NaN_present) \
        if(tensor.getLength() >= PARALLEL_THRESHOLD || tensor.getSize() >= PARALLEL_THRESHOLD)
    for (unsigned int index = 0; index < tensor.size_ * tensor.length_; index++)
        if (std::isnan(tensor.elements_[index]))
            is_NaN_present = true;


    return is_NaN_present;

}