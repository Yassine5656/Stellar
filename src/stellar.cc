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

#include <iostream>

#include "core/tensor/init.hh"
#include "core/tensor/io.hh"
#include "core/tensor/misc.hh"
#include "core/tensor/tensor.hh"

#define CLEAR_CONSOLE() std::cout << "\033[2J\033[H\n";

const std::string logo = "\n\n"
                         "\t\t\t\t\t\t\t\t $$$$$$\\    $$\\               $$\\ "
                         "$$\\                     \n"
                         "\t\t\t\t\t\t\t\t$$  __$$\\   $$ |              $$ "
                         "|$$ |                    \n"
                         "\t\t\t\t\t\t\t\t$$ /  \\__|$$$$$$\\    $$$$$$\\  $$ "
                         "|$$ | $$$$$$\\   $$$$$$\\  \n"
                         "\t\t\t\t\t\t\t\t\\$$$$$$\\  \\_$$  _|  $$  __$$\\ $$ "
                         "|$$ | \\____$$\\ $$  __$$\\ \n"
                         "\t\t\t\t\t\t\t\t \\____$$\\   $$ |    $$$$$$$$ |$$ "
                         "|$$ | $$$$$$$ |$$ |  \\__|\n"
                         "\t\t\t\t\t\t\t\t$$\\   $$ |  $$ |$$\\ $$   ____|$$ "
                         "|$$ |$$  __$$ |$$ |      \n"
                         "\t\t\t\t\t\t\t\t\\$$$$$$  |  \\$$$$  |\\$$$$$$$\\ $$ "
                         "|$$ |\\$$$$$$$ |$$ |      \n"
                         "\t\t\t\t\t\t\t\t \\______/    \\____/  "
                         "\\_______|\\__|\\__| \\_______|\\__|      \n\n\n\n";

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
            using namespace stellar::core;
            tensor::Tensor tensor(2050, 2050);
            tensor::init::fill_uniform(tensor, -5.5, 18.2, 2);
            tensor::misc::dump(tensor);
            tensor::io::save(tensor, "/tmp/tensor.bin");
            break;
        }
        case 2:
        {
            CLEAR_CONSOLE();
            using namespace stellar::core;
            tensor::Tensor tensor_saved(2050, 2050);
            tensor::io::load_(tensor_saved, "/tmp/tensor.bin");
            tensor::misc::dump(tensor_saved);
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
