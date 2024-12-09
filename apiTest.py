from sympy.core.benchmarks.bench_arit import timeit_Add_x05

from contextoBot import ContextoSolver
import random


import logging

if __name__ == "__main__":
    todaysGame = 15
    solver = ContextoSolver(webDriver="API", gameNumber=todaysGame)
    print("Created solver")
    # Select a random game using the APIDriver's built-in functionality
    solver.driver.selectGameByGameNumber()  # This will set a random game number
    try:
        print(f"Beginning game #{todaysGame}")
        guesses = solver.play_game()
        print(f"Game finished in {guesses} guesses")
    except Exception as e:
        logging.error(f"Error during game: {e}")
    finally:
        solver.log("Contexto solver", guesses)
        solver.cleanup()
