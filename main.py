import logging
from gloveTest import ContextoSolver1  # Import the updated ContextoSolver class
from gloveTest import ContextoSolver2

if __name__ == "__main__":
    # Initialize the solver
    solver = ContextoSolver2()

    try:
        # Run the game-playing logic
        guesses = solver.play_game()
        print(f"Game finished in {guesses} guesses.")
    except Exception as e:
        logging.error(f"Error during game: {e}")
    finally:
        # Ensure resources are cleaned up
        solver.cleanup()