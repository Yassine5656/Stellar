//
// Created by yassine on 2/12/26.
//

#include "tensor.h"

#include <iomanip>
#include <iostream>

#define MAX_ELM_RENDERED ((unsigned int) 3)
#define PADDING 10
#define PRECISION_RENDERED 4

void stellar::Tensor::fill_triangular(const scalar_t scalar,
                                      const bool upper_section) {
    // Assert square matrix
    assert(size_ == length_);
    if (upper_section) {
        for (unsigned int row = 0; row < size_; row++) {
            for (unsigned int col = 0; col < row; col++)
                elements_[row * length_ + col] = 0;
            for (unsigned int col = row; col <= size_; col++)
                elements_[row * length_ + col] = scalar;
        }
    } else {
        for (unsigned int row = 0; row < size_; row++) {
            for (unsigned int col = 0; col <= row; col++)
                elements_[row * length_ + col] = scalar;
            for (unsigned int col = row + 1; col < size_; col++)
                elements_[row * length_ + col] = 0;
        }
    }
}

void stellar::Tensor::dump() const {
    const auto backup_flags = std::cout.flags();
    std::cout << std::setprecision(PRECISION_RENDERED);
    std::cout << "Tensor: [" << size_ << " x " << length_ << "]" << std::endl;
    const unsigned int rows_rendered = std::min(MAX_ELM_RENDERED, size_);
    const unsigned int cols_rendered = std::min(MAX_ELM_RENDERED, length_);
    std::cout << "[" << std::endl;
    for (unsigned int row = 0; row < rows_rendered; row++) {
        std::cout << std::setw(PADDING) << " " << elements_[row * length_];
        unsigned int col = 1;
        for (; col < cols_rendered; col++) {
            std::cout << ", " << std::setw(PADDING) << elements_[row * length_ + col];
        }
        if (length_ - cols_rendered > col) {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        for (col = std::max(col, length_ - cols_rendered); col < length_; col
             ++) {
            std::cout << ", " << std::setw(PADDING) << elements_[row * length_ + col];
        }
        std::cout << std::endl;
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
        std::cout << std::endl;
    }

    for (unsigned int row = std::max(rows_rendered, size_ - rows_rendered);
         row < size_; row++) {
        std::cout << std::setw(PADDING) << " " << elements_[row * length_];
        unsigned int col = 1;
        for (; col < cols_rendered; col++) {
            std::cout << ", " << std::setw(PADDING) << elements_[row * length_ + col];
        }
        if (length_ - cols_rendered > col) {
            std::cout << ", " << std::setw(PADDING) << "...";
        }
        for (col = std::max(col, length_ - cols_rendered); col < length_; col
             ++) {
            std::cout << ", " << std::setw(PADDING) << elements_[row * length_ + col];
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;

    /*
    std::cout << "Min: " << std::setw(PADDING) << min(*this) << std::endl;
    std::cout << "Max: " << max(*this) << std::endl;
    std::cout << "Mean: " << mean(*this) << std::endl;
    std::cout << "NaN Present: " << check_invalid(*this) << std::endl;
    */
    std::cout.flags(backup_flags);
}
