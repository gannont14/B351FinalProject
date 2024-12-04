from contextoBot import ContextoSolver
import logging

if __name__ == "__main__":
    solver = ContextoSolver()
    try:
        guesses = solver.play_game()
        print(f"Game finished in {guesses} guesses")
    except Exception as e:
        logging.error(f"Error during game: {e}")
    finally:
        solver.cleanup()
