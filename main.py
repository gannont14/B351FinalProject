import logging
from gloveTest import ContextoSolver1  # Import the updated ContextoSolver class
from gloveTest import ContextoSolver2

if __name__ == "__main__":
    while True:
        print("Starting")
        # Initialize the solver
        solver = ContextoSolver2()
        print("Initialized")

        try:
            # Run the game-playing logic
            guesses = solver.play_game()
            print(f"Game finished in {guesses} guesses.")
            with open("oldVectorValues.csv", 'a') as fd:
                fd.write(f"{solver.driver.getCurrentGameNumber()},{guesses}\n")
        except Exception as e:
            logging.error(f"Error during game: {e}")
            with open("oldVectorValues.csv", "a") as fd:
                fd.write(f"{solver.driver.getCurrentGameNumber()},DNS\n")
        finally:
            # Ensure resources are cleaned up
            solver.cleanup()
