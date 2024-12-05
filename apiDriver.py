import requests


class APIDriver:
    def __init__(self, gameNumber=808):
        self.baseURL = "https://api.contexto.me/machado/en/game/"
        self.gameNum = gameNumber  # This is the game for 12/4, could add logic to increment every day but seems useless

    def makeGuess(self, word):
        url = f"{self.baseURL}{str(self.gameNum)}/{str(word)}"
        # Error
        response = requests.get(url)
        if response.status_code == 200:
            # Success
            data = response.json()
            return data

        # Error
        return None


if __name__ == "__main__":
    guesses = ["person", "place", "thing", "idea"]
    d = APIDriver()
    for guess in guesses:
        res = d.makeGuess(guess)
        print(res)
