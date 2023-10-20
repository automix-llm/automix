## AutoMix Library

### Installation

```bash
pip install automix-llm
```

### How to Use?

#### Step 1: Intialization

```python
from automix import Threshold, POMDP, Automix

meta_verifier = Threshold(num_bins = 8)   # Number of bins for discretization
mixer = Automix(meta_verifier, slm_column = 'llama13b_f1', llm_column = 'llama70b_f1', verifier_column = 'p_ver_13b', costs = [1, 50], verifier_cost = 1, verbose = False)
```

#### Step 2: Training
    
```python
mixer.train(train_data)
```

`train_data' should be pandas data frame with the metric scores for both slm and llm along with verifier confidence.

#### Step 3: Inference

```python
mixer.infer(test_row) # Returns True if query needs to be routed to LLM
```

or

```python
mixer.evaluate(test_data) # Performs Complete Evaluation on Test Data
```


### Variants of Meta-Verifier:

You can use fferent Varaints of Meta-Verifiers:
- Threshold: Find a single threshold to route queries to LLM.
- DoubleThreshold: Find two thresholds to route queries to LLM.
- GreedyPOMDP: A greedy approach based POMDP, that discretizes observation space
- POMDP: Use a POMDP to route queries to LLM.
- AutomixUnion: Use a union of a set of other meta-verifiers to route queries to LLM. During training it automatically selects the best meta-verifier.