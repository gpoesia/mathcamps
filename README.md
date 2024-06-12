# MathCAMPS - A dataset of mathematical problems synthesized from an educational curriculum

## Overview 

MathCAMPS is a dataset of synthetic math word problems derived from the [Mathematics Common Core](https://www.thecorestandards.org/Math/), a widely used curriculum in schools in the US. The Common Core contains a collection of *standards* for each grade (K-8, plus high school), each describing a specific mathematical ability that students should learn at that grade. Every problem in MathCAMPS is tied to a particular standard from grades K-8, allowing a detailed evaluation and analysis of mathematical skills in language models.

### How is this different from GSM8K or other datasets?

With chain-of-thought prompting, GPT-3 scored 51.3% on GSM8K. GPT-4 was much better: 87.1%. What did GPT-4 get better on that accounts for this? What was still hard? These questions are hard to answer with GSM8K and other datasets, which only directly offer aggregate accuracies. In contrast, in MathCAMPS, every problem is generated targeting a particular mathematical skill described in a widely used educational curriculum -- the Common Core Standards. This enables a much more detailed evaluation of specific mathematical abilities and challenges. Moreover:

* **Synthetic**: we can generate novel problems on demand, at scale, from any standard, ensuring no test-set contamination. In contrast, with other publicly available datasets, we can't really know.
* **Extensible**: we plan to extend the scope of the dataset over time by covering an increasing number of Common Core standards. This will include problems with diagrams, requiring multi-modal reasoning.
* **Follow-up questions**: besides word problems, our pipeline can also synthesize *follow-up questions* to each problem, allowing for an evaluation of language models in a "mathematical dialogue" setting: after being given a problem and receiving a response, we can ask a follow-up question that either modifies the original problem (*counterfactual follow-up*) or adds something to it (*incremental follow-up*), probing for deeper understanding. Our results show that many models, particular smaller ones, do not reliably answer these questions.

## Problems and model responses

MathCAMPS v1.0 contains 4900 original problems, and a total of 4707 follow-up problems from 44 distinct Common Core standards. The problems can be accessed in `problems/v1/`. This directory contains one JSON file for each standard (but 49 JSON files in total, since we have broken down some standards into multiple during generation).

Each JSON file has an *array* of *problem sequences* (with each such *sequence** being an array of objects representing individual problems). The first problem in a sequence is the "original problem", while subsequent problems are follow-ups (between 0 and 2 follow-ups, depending on the standard). For now, follow-ups are not cumulative - in cases where there are two of them, they both follow up on the original problem, not on each other. Here is an example from standard `3.OA.A.3` ("Use multiplication and division within 100 to solve word problems in situations involving equal groups, arrays, and measurement quantities"):

``` json
[
  [
    {
      "id": "3.OA.A.3-2-0",
      "standard": "3.OA.A.3",
      "symbolic-struct": "[[var x = 12]]\n[[var s = (8 * x)]]\n[[question f = ['s']]]\ntheme: Chair",
      "statement": "John has 12 tables. Each table requires 8 chairs. How many chairs does John need to accommodate all the tables?",
      "new symbolic struct": "[[var tables = 12]]\n[[var chairs_per_table = 8]]\n[[var total_chairs = tables * chairs_per_table]]\n[[question result = ['total_chairs']]]",
      "answer": "96",
      "tag": "original problem"
    },
    {
      "id": "3.OA.A.3-2-2",
      "standard": "3.OA.A.3",
      "symbolic-struct": "[[var x = 2]]\n[[var s = (8 * x)]]\n[[question f = ['s']]]\ntheme: Chair",
      "statement": "Suppose now, John only has 2 tables instead of 12. Using the same number of chairs per table, how many chairs would John need now to accommodate these tables?",
      "new symbolic struct": "[[var tables = 2]]\n[[var chairs_per_table = 8]]\n[[var total_chairs = tables * chairs_per_table]]\n[[question result = ['total_chairs']]]",
      "answer": "16",
      "tag": "modified information follow up"
    }
  ],
...

Here, *statement* has the word problem statement, and answer has the expected answer. The symbolic structure is what was sampled from our grammar encoding Common Core standards.


## Changelog

### v1.0.0

- Initial release of the dataset, problem generation pipeline and model outputs
