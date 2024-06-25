# MathCAMPS - A dataset of mathematical problems synthesized from an educational curriculum

## Overview 

MathCAMPS is a dataset of synthetic math word problems derived from the [Mathematics Common Core](https://www.thecorestandards.org/Math/), a widely used curriculum in schools in the US. The Common Core contains a collection of *standards* for each grade (K-8, plus high school), each describing a specific mathematical ability that students should learn at that grade. Every problem in MathCAMPS is tied to a particular standard from grades K-8, allowing a detailed evaluation and analysis of mathematical skills in language models.

### How is this different from GSM8K or other datasets?

With chain-of-thought prompting, GPT-3 scored 51.3% on GSM8K. GPT-4 was much better: 87.1%. What did GPT-4 get better on that accounts for this? What was still hard? These questions are hard to answer with GSM8K and other datasets, which only directly offer aggregate accuracies. In contrast, in MathCAMPS, every problem is generated targeting a particular mathematical skill described in a widely used educational curriculum -- the Common Core Standards. This enables a much more detailed evaluation of specific mathematical abilities and challenges. Moreover:

* **Synthetic**: we can generate novel problems on demand, at scale, from any standard, ensuring no test-set contamination. In contrast, with other publicly available datasets, we can't really know.
* **Extensible**: we plan to extend the scope of the dataset over time by covering an increasing number of Common Core standards. This will include problems with diagrams, requiring multi-modal reasoning.
* **Follow-up questions**: besides word problems, our pipeline can also synthesize *follow-up questions* to each problem, allowing for an evaluation of language models in a "mathematical dialogue" setting: after being given a problem and receiving a response, we can ask a follow-up question that either modifies the original problem (*counterfactual follow-up*) or adds something to it (*incremental follow-up*), probing for deeper understanding. Our results show that many models, particular smaller ones, do not reliably answer these questions.

## Problems and model responses

MathCAMPS v1.0 contains 9707 total problems (4900 original problems, and a total of 4707 follow-up problems) from 44 distinct Common Core standards. The problems can be found in `problems/v1/mathcamps.json`. This directory also contains one JSON file for each standard (but 49 JSON files in addition to `mathcamps.json`, since we have broken down some standards into multiple during generation). The standard-specific files also have the symbolic structure associated with problems, which were stripped in `mathcamps.json` for simplicity.

`problems/v1/mathcamps.json` is a JSON file with a single array of problems. Each problem is an object with the following fields:

```json
{
  "id": "<<unique problem identifier>>",
  "standard": "<<ID of the Common Core standard this problem belongs to>>",
  "statement": "<<problem statement, in natural language>>",
  "answer": "<<expected final answer>>",
  "type": "<<either 'original-problem', 'incremental-follow-up' or 'counterfactual-followup'>>",
  "followup-to": "<<if type is not 'original-followup', then the ID of the problem this one follow-up on. Otherwise null>>"
},
```

For example:

```json
{
  "id": "2.MD.C.8-0-0",
  "standard": "2.MD.C.8",
  "statement": "Liam had $90 in twenties, tens and fives. He spent $81 on a new video game. How much money in dollars does Liam have left?",
  "answer": "9",
  "type": "original-problem",
  "followup_to": null
},
```

You can also load the dataset from the Hugging Face datasets hub:

```python
import datasets
mathcamps = datasets.load_dataset('mathcamps/mathcamps')
print(mathcamps['train'][1000]['statement'])
```

The model responses to these problems are under `model-responses/v1/`. These contain more than 220K LLM responses to all problems, across 21 models we evaluated on MathCAMPS (including GPT-4o, GPT-3.5 Turbo, all Claude 3 models, all LLaMA-3 models, etc), along with the extracted final answer from the model's response, the ground-truth answer, and whether the final answer was correct (which is obtained with a semantic comparison, e.g. equivalent fractions are considered equal).

## Changelog

### v1.0.0

- Initial release of the dataset, problem generation pipeline and model outputs

### v1.0.1

- Release of `mathcamps.json` and Hugging Face dataset

