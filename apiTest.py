from contextoBot import ContextoSolver


import logging

if __name__ == "__main__":
    todaysGame = 809
    solver = ContextoSolver(webDriver="API", gameNumber=todaysGame)
    print("Created solver")
    try:
        guesses = solver.play_game()
        print(f"Game finished in {guesses} guesses")
    except Exception as e:
        logging.error(f"Error during game: {e}")
    finally:
        solver.log("Contexto solver", guesses)
        solver.cleanup()
