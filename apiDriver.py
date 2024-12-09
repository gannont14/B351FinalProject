import requests
import random
from datetime import datetime
import json


class APIDriver:
    def __init__(self, gameNumber=809):
        self.baseURL = "https://api.contexto.me/machado/en/game/"
        self.gameNum = gameNumber  # This is the game for 12/4, could add logic to increment every day but seems useless
        self.gameOver = False

    # Helper function that just calls the api and returns the data
    def makeGuess(self, word):
        url = f"{self.baseURL}{str(self.gameNum)}/{str(word)}"
        # Error
        response = requests.get(url)
        if response.status_code == 200:
            # Success
            data = response.json()

            # check if that was the right answe,r if so, set game over
            if data['distance'] == 0:
                self.gameOver = True
            print(data)
            return data

        # Error
        return None

    def guessWord(self, word):
        # score returned from the API is one less that is displayed on the site
        data = self.makeGuess(word)
        if data:
            return data['word'], data['distance'] + 1
        # All of this can be updatead to get the Lemma from this
        return None, None

    def selectGameByGameNumber(self, gameNum=None):
        if gameNum is None:
            gameNum = random.randint(1, self.gameNumber)

        self.gameNumber = gameNum

    # This is also kind of redundant, but want it to interface well
    def checkIfGameOver(self):
        return self.gameOver

    # doesn't do anything but won't interface well if i don't add this
    # ... actually need to add logging to keep track of score for the game, just append to a file
    def quitDriver(self):
        # log the amount of guesses
        print("Quitting driver")

    # def log(self, logTitle, guesses):
    #     with open('log.txt', 'a') as f:
    def log(self, title, num_guesses):
        log_data = {
            'game_number': self.gameNum,
            'title': title,
            'total_guesses': num_guesses
            # 'guesses': self.game_log
        }

        filename = f"game_logs/game_{self.gameNum}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)  # f.write(f"{logTitle} - {guesses} guesses")


if __name__ == "__main__":
    guesses = ["person", "place", "thing", "idea"]
    d = APIDriver()
    for guess in guesses:
        res = d.makeGuess(guess)
        print(res)
