import time


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

import heapq
"""
    Driver for contexto, to run, must first activate virtual environment
    and then have the correct web drivers installed
    https://selenium-python.readthedocs.io/installation.html
    """


class Driver():
    """
    Constructor: create driver, if you don't input any
    parameters will default to chrome
    """

    def __init__(self, driverType="Chrome"):
        # Creates driver based on browser
        # Could be updated to max heap instead of set
        self.guesses = {}
        # heapq library is only a min heap, set values to negative before inserting them, and
        # then use absolute value when popping them
        self.guessesHeap = []
        self.numGuesses = 0
        self.driver = None
        print(f"Initing with {driverType} drive Type")
        if driverType == "Chrome":
            self.driver = webdriver.Chrome()
        elif driverType == "Firefox":
            self.driver = webdriver.Firefox()
        elif driverType == "Edge":
            self.driver = webdriver.Edge()
        elif driverType == "Safari":
            self.driver = webdriver.Safari()
        else:
            print("Ivalid Driver type")
            return


# Fetches site from url
        siteURL = "https://contexto.me/"
        self.driver.get(siteURL)

    """
        Guesses a word based on the string input, this is just for a single
        word
        Return: A tuple with the results of the guess,
                if returns None for either
                portion of tuple, it was an invalid guess, and
                the input box was cleared
        """

    def guessWord(self, word):
        # Find the form input box by name , is named 'word' on site
        wordBoxDomName = "word"
        inputElem = self.driver.find_element(By.NAME, wordBoxDomName)
        inputElem.send_keys(word, Keys.ENTER)
        time.sleep(2)
        # get the most recent words score and value
        recentGuessScore, recentGuessWord = self.getRecentGuess()
        print(f"Recent: {recentGuessWord}")
        # if the guess is 'Too common' the getRecentGuess will instead return 'None', as it is catching
        # the NoSuchElementException, if it is none, the input text box needs to be cleared so it doesn't mess with the next word
        if recentGuessWord is None or recentGuessScore is None:
            # need to clear the intpu
            inputElem.clear()
        else:
            heapq.heappush(self.guessesHeap,
                           (recentGuessScore, recentGuessWord))
            self.numGuesses += 1

        return (recentGuessWord, recentGuessScore)

    """
        The same function as above, except takes a list of words, mostly will be used for testing
        """

    def guessWords(self, words):
        if len(words) < 1:
            print("invalid list")
            return
        for word in words:
            self.guessWord(word)
        """
        Gets all of the current guesses so far, stores in set 'self.guesses', where the key is the word as a string,
        and the score is the value as an int
        """

    def getAllGuesses(self):
        # Site is comprised of 'guess-history' -> which contains all guess, each guess
        # is in a 'row-wrapper' div, and one 'row-wrapper current', each row wrapper has a row
        # inside, which contains two spans, being the guess and the score
        guessHistoryElemName = 'guess-history'
        guessHistoryElem = self.driver.find_element(
            By.CLASS_NAME, guessHistoryElemName)
        rows = guessHistoryElem.find_elements(By.CLASS_NAME, 'row')
        # Each row has two spans in it, the first being the guess and the
        # second being the score, add these both to the guess dictionary
        for row in rows:
            rowSpans = row.find_elements(By.TAG_NAME, 'span')
            # first one is the guess,
            guessText = rowSpans[0].text
            # Second one is the score
            guessScore = int(rowSpans[1].text)
            # Add to the dict if not already in it
            if guessText not in self.guesses:
                self.guesses[guessText] = guessScore
        """
        Gets the most recent guess, which is  the top guess on the website, as well as the score, will return (None,None)
        if the find_elements throws an NoSuchElementException
        """

    def getRecentGuess(self):
        # In the div with messages className, there is a row for the mosot recent guess
        try:
            messagesClassName = 'message'
            messagesDiv = self.driver.find_element(
                By.CLASS_NAME, messagesClassName)
            # Get the row in this div
            recentGuessRow = messagesDiv.find_element(By.CLASS_NAME, "row")
            # Get the two spans within that class
            spans = recentGuessRow.find_elements(By.TAG_NAME, "span")
            # first one is the guess,
            guessText = spans[0].text
            # Second one is the score
            guessScore = int(spans[1].text)
        except NoSuchElementException:
            print("element doesn't exist")
            return (None, None)

        return (guessScore, guessText)

    """
    Returns boolean of it game is over, can check the guessCount for the amount that it took
    """

    def checkIfGameOver(self):

        try:
            chartClassName = 'chart-wrapper'
            # Tries to find the classname for the wrapper, if it doesn't, it
            # will throw expection If exception is caught, will return false
            _ = self.driver.find_element(By.CLASS_NAME, chartClassName)
            return True
        except NoSuchElementException:
            return False

    def getBestGuess(self):
        if len(self.guessesHeap) == 0:
            print("Heap is empty")
            return -1

        # pop off of the heapq, and set value back to positive
        guessScore, guessText = heapq.heappop(self.guessesHeap)

        return (guessScore, guessText)

    """
        Helpers:
        printGuesses: print the set of all guesses
        quitDriver: just called driver.quit
    """

    def printGuesses(self):
        print(f"Guesses: {self.guesses}")

    def quitDriver(self):
        self.driver.quit()


if __name__ == "__main__":
    guesses = ["Hello", "again", "test", "boat", "table"]
    contDriver = Driver("Firefox")
    contDriver.guessWords(guesses)
    print(f"Heap: {contDriver.guessesHeap}")
    contDriver.getAllGuesses()
    contDriver.printGuesses()
    print(f"Best guess: {contDriver.getBestGuess()}")
    print(contDriver.numGuesses)

    # print(f"isGameOver: {contDriver.checkIfGameOver()}")
    # contDriver.guessWord("counter")
    # print(f"isGameOver: {contDriver.checkIfGameOver()}")
