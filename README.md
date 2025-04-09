# uhhc-toolbox

This toolbox provides functionalities for the generation and validation of instances related to the Unified Home Healthcare Routing and Scheduling Problem (UHHRSP).

The UHHRSP is a complex optimization problem that involves planning and scheduling home healthcare services, considering constraints such as caregiver availability, patient requirements, time windows, and routing logistics.

Key features of this toolbox include:

- Instance generation: Tools to create problem instances with customizable parameters such as number of patients, caregivers, and geographical distribution.
- Validation: Methods to ensure that generated instances adhere to the problem's constraints and are suitable for testing optimization algorithms.
- Flexibility: Support for various configurations and extensions of the UHHRSP to accommodate different real-world scenarios.

This toolbox is intended for researchers and practitioners working on optimization problems in the domain of home healthcare logistics.

## Command Line Usage

The `uhhc-toolbox` provides a command-line interface (CLI) for ease of use. Below are the available commands and their descriptions:

### Help

For detailed help on any command, use:
```bash
uhhc-toolbox <command> --help
```

## Installation Instructions

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install the toolbox, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/iolab-uniud/uhhc-toolbox.git
    cd uhhc-toolbox
    ```

2. Install the dependencies:
    ```bash
    poetry install
    ```

3. If you need the instance generator functionality, which requires `osrm-backend`, install with extras:
    ```bash
    poetry install --with generator
    ```

    **Note:** Ensure that `osrm-backend` is installed and properly configured on your system before using the generator.

You are now ready to use the `uhhc-toolbox`!