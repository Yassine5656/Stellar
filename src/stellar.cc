//
// Created by yassine on 2/20/26.
//

#include <chrono>
#include <iostream>

#include "../include/core/tensor.hh"

#define CLEAR_CONSOLE() std::cout << "\033[2J\033[H\n";

const std::string logo =
        "\n\n"
        "\t\t\t\t\t\t\t\t $$$$$$\\    $$\\               $$\\ $$\\                     \n"
        "\t\t\t\t\t\t\t\t$$  __$$\\   $$ |              $$ |$$ |                    \n"
        "\t\t\t\t\t\t\t\t$$ /  \\__|$$$$$$\\    $$$$$$\\  $$ |$$ | $$$$$$\\   $$$$$$\\  \n"
        "\t\t\t\t\t\t\t\t\\$$$$$$\\  \\_$$  _|  $$  __$$\\ $$ |$$ | \\____$$\\ $$  __$$\\ \n"
        "\t\t\t\t\t\t\t\t \\____$$\\   $$ |    $$$$$$$$ |$$ |$$ | $$$$$$$ |$$ |  \\__|\n"
        "\t\t\t\t\t\t\t\t$$\\   $$ |  $$ |$$\\ $$   ____|$$ |$$ |$$  __$$ |$$ |      \n"
        "\t\t\t\t\t\t\t\t\\$$$$$$  |  \\$$$$  |\\$$$$$$$\\ $$ |$$ |\\$$$$$$$ |$$ |      \n"
        "\t\t\t\t\t\t\t\t \\______/    \\____/  \\_______|\\__|\\__| \\_______|\\__|      \n\n\n\n";

const std::string menu = "Stellar menu:\n"
        "\t1.\n"
        "\t2. Launch testsuite\n"
        "\t3. Exit\n"
        "\n"
        "Choose a choice among the above list: ";

const std::string error_menu =
        "The choice must be a valid number among the menu list.\n";

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    // TODO: May use arguments to set parameters for debug or other purposes
    // TODO: Set a logic for asking what the user want to do
    bool is_running = true;
    std::string input;
    /*
    const auto start_global = std::chrono::high_resolution_clock::now();
    auto A = stellar::core::Tensor(10000, 500);
    A.fill_sequence(-2, 2);
    A.dump();
    auto B = stellar::core::Tensor(500, 5000);
    B.fill_sequence(-4, 4);
    B.dump();
    const auto start_multiply = std::chrono::high_resolution_clock::now();
    A.multiply(B);
    const auto end_multiply = std::chrono::high_resolution_clock::now();
    A.dump();
    const auto end_global = std::chrono::high_resolution_clock::now();
    std::cout << "Multiply Time: " << std::chrono::duration<double, std::milli>(end_multiply - start_multiply) << std::endl;
    std::cout << "Global time: " << std::chrono::duration<double, std::milli>(end_global - start_global) << std::endl;
    return 0;
    auto matrix1= stellar::Tensor(5000, 5000);
    matrix1.test_move();
    const auto start_global = std::chrono::high_resolution_clock::now();
    const auto start_init = std::chrono::high_resolution_clock::now();
    auto matrix = stellar::Tensor(15000, 15000);
    const auto end_init = std::chrono::high_resolution_clock::now();
    const auto start_init_copy = std::chrono::high_resolution_clock::now();
    auto m = stellar::Tensor(matrix);
    const auto end_init_copy = std::chrono::high_resolution_clock::now();
    const auto start_fill = std::chrono::high_resolution_clock::now();
    matrix.fill_uniform(-144, 144, 4);
    const auto end_fill = std::chrono::high_resolution_clock::now();
    matrix.set(0, 0, NAN);
    const auto start = std::chrono::high_resolution_clock::now();
    matrix.dump();
    const auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Init Time: " << std::chrono::duration<double, std::milli>(end_init - start_init) << std::endl;
    std::cout << "Init Copy Time: " << std::chrono::duration<double, std::milli>(end_init_copy - start_init_copy) << std::endl;
    std::cout << "Fill Time: " << std::chrono::duration<double, std::milli>(end_fill - start_fill) << std::endl;
    std::cout << "Dump Time: " << std::chrono::duration<double, std::milli>(end - start) << std::endl;
    std::cout << "Global time: " << std::chrono::duration<double, std::milli>(end - start_global) << std::endl;
    */
    std::cout << logo;
    while (is_running) {
        std::cout << menu;
        std::cin >> input;
        switch (strtol(input.c_str(), nullptr, 10)) {
            case 1: {
                CLEAR_CONSOLE();
                auto tensor = stellar::core::Tensor(30000, 41000);
                tensor.fill_uniform(-5.5, 18.2, 2);
                tensor.dump();
                tensor.save_to_binary("/tmp/clembibi.bin");
                //auto tensor_saved = stellar::core::Tensor(3, 4);
                //tensor_saved.fill_from_binary("/tmp/clembibi.bin");
                //tensor_saved.dump();
                break;
            }
            case 2: {
                CLEAR_CONSOLE();
                break;
            }
            case 3: {
                std::cout << "\nExit application...\n";
                is_running = false;
                break;
            }
            default:
                std::cout << error_menu << std::endl;
        }
    }

    return 0;
}
