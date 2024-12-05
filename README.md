
# ContextoSolver

**ContextoSolver** is a Python program designed to solve word-guessing games by leveraging GloVe embeddings and an intelligent seed-based word-ranking system. It features two implementations: `ContextoSolver1` (new version) and `ContextoSolver2` (old version). Each class attempts to predict the target word by analyzing similarity rankings and optimizing guesses through machine learning models.

---

## **Features**
1. **GloVe Embeddings:**
   - Both solvers use the `glove-wiki-gigaword-300` pre-trained model for word vectorization.

2. **Seed-Based Guessing:**
   - Guesses are generated based on a combination of positive and negative seeds, selected dynamically from previous guesses.

3. **Dynamic Seeds Adjustment:**
   - The number of seeds (`num_seeds`) is adjusted based on the total number of guesses:
     - **3 seeds:** Initial phase.
     - **5 seeds:** After 10 guesses.
     - **10 seeds:** After 30 guesses.

4. **Game Selection:**
   - By default, the program selects a random game, but specific game numbers can be provided during initialization.

5. **Word List:**
   - A comprehensive word list is loaded using NLTK, with fallback to the GloVe vocabulary in case of errors.

---

## **Key Differences Between ContextoSolver1 and ContextoSolver2**

| Feature                     | **ContextoSolver1 (New)**                              | **ContextoSolver2 (Old)**                              |
|-----------------------------|-------------------------------------------------------|-------------------------------------------------------|
| **Weighting System**         | Implements a normalized weighting system based on scores to adjust the influence of positive/negative seeds. | Relies on static positive/negative seeds without score-based weighting. |
| **Handling Positive Seeds** | Dynamically weights positive seeds (lower scores are prioritized). | Equal weight for all positive seeds.                 |
| **Negative Seeds**           | Uses normalized weights for negative seeds, adjusted similarly to positive seeds. | No weighting; uses static list of negative seeds.     |
| **Performance**              | May perform better in some games due to dynamic weights but is prone to getting stuck if weighting is not optimized. | More stable across games but lacks flexibility in adapting to specific scenarios. |
| **Fallback Logic**           | Simplified handling; exits cleanly if no valid candidates are found. | Implements fallback to a predefined list of common words if guesses fail. |

---

## **Challenges with ContextoSolver1**
- **Weighting Optimization:** While the dynamic weighting system can improve accuracy, it sometimes causes the model to get stuck. Further tuning is required to balance the influence of high-scoring words.
- **Performance Variability:** Depending on the game, the new weighting system may either outperform or underperform compared to `ContextoSolver2`.

---

## **Installation**

1. Install the required dependencies:
   ```bash
   pip install numpy sklearn nltk gensim
   ```
2. Download the NLTK word list:
   ```python
   import nltk
   nltk.download('words')
   ```

---

## **Conclusion**
Both `ContextoSolver1` and `ContextoSolver2` are effective in solving word games, with distinct strengths and weaknesses. The new solver offers more dynamic behavior but requires further refinement to avoid getting stuck. Users should experiment with both solvers to determine the best fit for specific scenarios.
