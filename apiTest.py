from contextoBot import ContextoSolver
from datetime import datetime


import logging

if __name__ == "__main__":
    print(datetime.now().strftime('%Y%m%d_%H%M%S'))
    todaysGame = 809
    for gameNum in range(800, todaysGame):
        if gameNum == 801:
            continue
        solver = ContextoSolver(webDriver="API", gameNumber=gameNum)
        print(f"Created solver for game: {gameNum}")
        try:
            guesses = solver.play_game()
            print(f"Game finished in {guesses} guesses")
        except Exception as e:
            logging.error(f"Error during game: {e}")
        finally:
            solver.log("Contexto solver", guesses)
            solver.cleanup()
