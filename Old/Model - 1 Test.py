import os

# Define the path to the game numbers file
GAME_NUMBERS_FILE = "Model 1 Game Numbers.txt"

# Define the path to the ContextoSolver script
CONTEXTO_SOLVER_SCRIPT = "contextoBot - Model 1.py"


def get_next_game_number():
    """Get the next game number from the file and remove it."""
    if not os.path.exists(GAME_NUMBERS_FILE):
        print("Game numbers file not found!")
        return None

    # Read the file
    with open(GAME_NUMBERS_FILE, "r") as file:
        lines = file.readlines()

    # Filter out comments and empty lines
    lines = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    if not lines:
        print("No more game numbers left to process.")
        return None

    # Get the first game number
    next_game = lines[0].strip()

    # Rewrite the file without the first game number
    with open(GAME_NUMBERS_FILE, "w") as file:
        file.writelines(line + "\n" for line in lines[1:])

    # Debugging: Print the remaining game numbers
    print(f"Remaining game numbers after removing {next_game}: {lines[1:]}")

    return next_game


def run_contexto_solver(game_number):
    """Run the ContextoSolver for a specific game number."""
    if game_number is None:
        return

    print(f"Starting ContextoSolver for game #{game_number}")
    print(f"Using script: {CONTEXTO_SOLVER_SCRIPT}")

    # Check if the script exists
    if not os.path.exists(CONTEXTO_SOLVER_SCRIPT):
        print(f"Error: Script file '{CONTEXTO_SOLVER_SCRIPT}' not found!")
        return

    # Call the ContextoSolver script with the game number
    os.system(f"python \"{CONTEXTO_SOLVER_SCRIPT}\" {game_number}")
    print(f"Finished ContextoSolver for game #{game_number}")


def main():
    while True:
        game_number = get_next_game_number()
        if game_number is None:
            break  # Stop if no more game numbers
        run_contexto_solver(game_number)

    print("All games have been processed!")


if __name__ == "__main__":
    main()