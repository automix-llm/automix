# AutoMix: Automatically Mixing Language Models


<div align="center">
    <img src="https://github.com/automix-llm/automix/assets/1304693/a81ba101-247d-4fa7-8dc3-989dd5884483" width="500">
</div>


## What is AutoMix?

The idea behind AutoMix is simple: 

1. Send a query to small language model (SLM), gets a noisy label on its correctness using few-shot self-verification done with the same model (SLM).

2. Use a meta-verifier to "double check" verifier's output, and route the query to a larger language model (LLM) if needed.



## Notebooks

### Running inference

- [**Step1 Run inference to solve tasks**](https://github.com/automix-llm/automix/blob/main/colabs/Step1_SolveQueries.ipynb) - Task prompts, code to run inference from different language models.
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automix-llm/automix/blob/main/colabs/Step1_SolveQueries.ipynb)

### Few-shot self-verification

- [**Step2 Self Verify**](https://github.com/automix-llm/automix/blob/main/colabs/Step2_SelfVerify.ipynb) - Verification prompts, code to run verification on the outputs produced in step 1.
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automix-llm/automix/blob/main/colabs/Step2_SelfVerify.ipynb)



### Meta-verification

- [**Step3 Meta Verify**](github.com/automix-llm/automix/blob/main/colabs/Step3_MetaVerify.ipynb) - Run meta-verification using different AutoMix methods on outputs produced from Step 2. 

- You can run `python setup.py install' to use the meta-verifier system wide.

- To replicate paper numbers, run `python scripts paper_results.py`


## Data and Outputs


- We experiment with 5 datasets: CNLI, CoQA, NarrativeQA, QASPER, and Quality.


- Note: The dataset are sourced from [scrolls](https://www.scrolls-benchmark.com/). Please cite scrolls and the appropriate sources if we use these datasets. We are making them available in a sinlge jsonl file for ease of use and reproducibility. For details on how CoQa was prepared, see [**Preparing COQA**](https://github.com/automix-llm/automix/blob/main/colabs/Preparing_COQA.ipynb).
   

- **Inputs:** All input data for the AutoMix project is provided in `automix_inputs.jsonl`. You can access and download it directly from [Google Drive](https://drive.google.com/file/d/1dhyt7UuYumk9Gae9eJ_mpTVrLeSTuRht/view?usp=sharing).

- **Outputs from LLAMA2:** The outputs generated using the LLAMA2 model are stored in `automix_llama2_outputs.jsonl`, available alongside the input file in the linked Google Drive.

```
id: A unique identifier for each question and answer pair.
pid: An additional identifier potentially mapping to specific instances or model variants.
base_ctx: The context.
question: Input question or query.
output: Ground truth.
dataset: .
llama13b_pred_ans: The answer generated by the llama13b model.
llama70b_pred_ans: The answer generated by the llama70b model.
llama13b_ver: Verification outputs of the llama13b model’s answers.
```

### Stats

```txt
dataset       split
cnli          train    7191
              val      1037
coqa          train    3941
              val      3908
narrative_qa  train    9946
              val      5826
qasper        train    2556
              val      1715
quality       train    2515
              val      2085
Name: split, dtype: int64
```







