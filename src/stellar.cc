//
// Created by yassine on 2/20/26.
//

#include <iostream>

#include "core/tensor.hh"

#define CLEAR_CONSOLE() std::cout << "\033[2J\033[H\n";

const std::string logo = "\n\n"
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

const std::string error_menu = "The choice must be a valid number among the menu list.\n";

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // TODO: May use arguments to set parameters for debug or other purposes
    // TODO: Set a logic for asking what the user want to do
    bool is_running = true;
    std::string input;
    std::cout << logo;
    while (is_running)
    {
        std::cout << menu;
        std::cin >> input;
        switch (strtol(input.c_str(), nullptr, 10))
        {
        case 1:
        {
            CLEAR_CONSOLE();
            auto tensor = stellar::core::Tensor(25000, 25000);
            tensor.fill_uniform(-5.5, 18.2, 2);
            tensor.dump();
            tensor.save_to_binary("/tmp/clembibi.bin");
            auto tensor_saved = stellar::core::Tensor(25000, 25000);
            tensor_saved.fill_from_binary("/tmp/clembibi.bin");
            tensor_saved.dump();
            break;
        }
        case 2:
        {
            CLEAR_CONSOLE();
            break;
        }
        case 3:
        {
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
