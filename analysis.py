#!/usr/bin/env python3


import collections
from dataclasses import dataclass
import grammar
from scipy.stats import pearsonr

import argparse
import json


TABLE_CONFIG_PATH = "config/main_table.json"


def translate_object(obj, translations):
    """Recursively replace values in a JSON object according to a dictionary."""
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            if key in translations:
                new_obj[translations[key]] = translate_object(value, translations)
            else:
                new_obj[key] = translate_object(value, translations)
        return new_obj
    elif isinstance(obj, list):
        return [translate_object(item, translations) for item in obj]
    elif isinstance(obj, str):
        return translations.get(obj, obj)
    else:
        return obj


def plot_vegalite(template: str, data: list, output_path: str, translations={}):
    with open(f'vega-lite/{template}.json') as f:
        spec = json.load(f)

    spec['data'] = {'values': data}
    spec = translate_object(spec, translations)

    with open(output_path + '.json', 'w') as f:
        json.dump(spec, f)


@dataclass
class Model:
    vendor: str
    model_id: str
    display_name: str
    category: str  # open or closed.


def load_results(results: list[str]) -> (dict, set, set):
    all_results = {}
    standards, grades = set(), set()
    for result in results:
        with open(result, 'r') as f:
            all_results.update(json.load(f))

    for result in all_results.values():
        standards.add(result['standard'])
        grades.add(result['standard'].split('.')[0])
    return all_results, standards, grades


def compute_accuracy(results: dict, model_id: str,
                     predicate,
                     followups=None) -> float:
    total = 0
    correct = 0
    for result in results.values():
        if result['model'] == model_id and predicate(result['standard'], result['problem'], None):
            total += 1
            if result['correct']:
                if not followups:
                    correct += 1
                else:
                    total_followups, total_followups_correct = 0, 0

                    for followup in result['followups'].values():
                        if not predicate(result['standard'], result['problem'], followup['problem']):
                            continue
                        total_followups += 1
                        if followup['correct']:
                            total_followups_correct += 1

                    if followups == 'all':
                        if total_followups == total_followups_correct:
                            correct += 1
                    elif followups == 'any':
                        if total_followups_correct > 0:
                            correct += 1
                    else:
                        raise ValueError(f'Invalid followups value: {followups} (should be all, any or None)')
    if not total:
        print('Warning: incomplete results for', model_id)
        return 0
    return correct / total


def get_followups_with_correct_answers(results: dict) -> (set, set):
    followup_ids = set()
    problem_ids = collections.defaultdict(set)
    standards_with_followup_ids = collections.defaultdict(set)

    for result in results.values():
        for followup in result.get('followups').values():
            if followup['correct']:
                followup_ids.add(followup['problem'])
                problem_ids[result['problem']].add(followup['problem'])
                standards_with_followup_ids[result['standard']].add(result['problem'])

    return followup_ids, problem_ids, standards_with_followup_ids


def rank_models(results: dict, model_ids: list[str], standard_predicate, followups=None) -> list[str]:
    ranking = []
    for model_id in model_ids:
        ranking.append((model_id, compute_accuracy(results, model_id, standard_predicate, followups)))
    ranking.sort(key=lambda x: x[1], reverse=True)
    return [model_id for model_id, _ in ranking]


def get_ordinal_str(n: int) -> str:
    if n == 1:
        return '$1^{st}$'
    elif n == 2:
        return '$2^{nd}$'
    elif n == 3:
        return '$3^{rd}$'
    else:
        return f'${n}^{{th}}$'


def generate_main_table(results):
    all_results, _standards, grades = load_results(results)
    print(len(all_results), 'results loaded')

    # Sort grades in ascending order, with 'K' coming first.
    grades = sorted(list(grades), key=lambda x: (x[0] != 'K', x))

    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    models = []

    for row in table_config['models']:
        models.append(Model(*row))

    columns = [('All', lambda standard, _p, _f: True)]

    for grade in grades:
        columns.append((grade, lambda standard, _p, _f: standard.startswith(grade + '.')))

    results_table = []

    for model in models:
        model_results = []
        for grade, predicate in columns:
            model_results.append(compute_accuracy(all_results,
                                                  model.model_id,
                                                  predicate))
        results_table.append(model_results)

    table_contents = []
    for model, model_results in zip(models, results_table):
        table_contents.append([model] + model_results)

    table_contents.sort(key=lambda x: (x[0].category, x[1]), reverse=True)

    headers = ['Vendor', 'Model'] + [column[0] for column in columns]

    latex_lines = []
    latex_lines.append(r'\begin{tabular}{c c|' + ' '.join(['c'] * len(columns)) + '}')
    latex_lines.append(r'\toprule')
    latex_lines.append(' & '.join(['\\textbf{' + header + '}' for header in headers]) + r'\\')
    last_category = None
    for model, *model_results in table_contents:
        if model.category != last_category:
            latex_lines.append(r'\midrule')
        last_category = model.category
        latex_lines.append(' & '.join([model.vendor, model.display_name] +
                                      [f'{result:.2f}' for result in model_results]) + r'\\')
    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')

    print('\n'.join(latex_lines))


def generate_followups_accuracy_drop_table(results: list[str], n=2):
    standards = grammar.load_standards().values()
    all_results, _, _ = load_results(results)

    followups_with_correct_answers, problems_with_followups, standards_with_followups = \
        get_followups_with_correct_answers(all_results)

    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    models = []

    for row in table_config['models']:
        models.append(Model(*row))

    overall_ranking = rank_models(all_results,
                                  [model.model_id for model in models],
                                  lambda _s, _p, _f: True)

    accuracy_on_all_followups = {
        model.model_id: compute_accuracy(all_results, model.model_id,
                                         lambda _s, _p, _f: _s in standards_with_followups,
                                         followups='all')
        for model in models
    }

    models.sort(key=lambda m: overall_ranking.index(m.model_id))

    # Filter out standards without followups.
    standards = [standard
                 for standard in standards
                 if standard.types_of_fup]

    standard_ids = [standard.id for standard in standards]

    accuracy_by_standard_no_followups = {}
    accuracy_by_standard_all_followups = {}

    for s in ['5.NBT.B.6', '6.NS.B.2']:
        standards_with_followups.pop(s)

    for standard in standard_ids:
        if standard not in standards_with_followups:
            continue

        accuracy_by_standard_no_followups[standard] = {}
        accuracy_by_standard_all_followups[standard] = {}

        for model in models:
            def standard_predicate(s, p, f):
                return (s == standard and
                        p in problems_with_followups and True)

            accuracy_by_standard_no_followups[standard][model.model_id] = \
                compute_accuracy(all_results, model.model_id,
                                 standard_predicate,
                                 followups=None)
            accuracy_by_standard_all_followups[standard][model.model_id] = \
                compute_accuracy(all_results, model.model_id,
                                 standard_predicate,
                                 followups='all')

    latex_lines = []
    latex_lines.append(r'\begin{tabular}{c p{4cm}' + ' c ' * n + '}')
    latex_lines.append(r'\toprule')
    latex_lines.append(rf'\textbf{{Model}} & Acc. w/ Follow-ups & \multicolumn{{{n}}}{{c}}{{\textbf{{Top {n} accuracy drops with follow-ups}}}} \\')

    last_category = None

    for model in models:
        if last_category != model.category:
            latex_lines.append(r'\midrule')
        last_category = model.category

        model_columns = [model.display_name, f'{accuracy_on_all_followups[model.model_id]:.2f}']

        accuracy_drops = []
        for standard in standard_ids:
            if standard not in standards_with_followups:
                continue

            no_followups_accuracy = accuracy_by_standard_no_followups[standard][model.model_id]
            all_followups_accuracy = accuracy_by_standard_all_followups[standard][model.model_id]
            accuracy_drop = no_followups_accuracy - all_followups_accuracy
            accuracy_drops.append((standard, no_followups_accuracy, all_followups_accuracy, accuracy_drop))

        accuracy_drops.sort(key=lambda x: x[3], reverse=True)

        for standard, no_followups_accuracy, all_followups_accuracy, accuracy_drop in accuracy_drops[:n]:
            model_columns.append(f'{standard} ({no_followups_accuracy:.2f} \\downto {all_followups_accuracy:.2f})')

        latex_lines.append(' & '.join(model_columns) + r'\\')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')

    print('\n'.join(latex_lines))


def generate_strengths_weaknesses_table(results: list[str], n=2):
    results, standards, grades = load_results(results)

    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    models = []
    for row in table_config['models']:
        models.append(Model(*row))

    overall_ranking = rank_models(results,
                                  [model.model_id for model in models],
                                  lambda _s, _p, _f: True)

    standard_rankings = {}

    for standard in standards:
        ranking = rank_models(results,
                              [model.model_id for model in models],
                              lambda s, _p, _f: s == standard)
        standard_rankings[standard] = ranking

    large_deviations_by_model = {}

    for m in models:
        deviations = []
        for standard in standards:
            ranking = standard_rankings[standard]
            original_ranking = overall_ranking.index(m.model_id) + 1
            standard_ranking = ranking.index(m.model_id) + 1
            deviations.append((standard, original_ranking, standard_ranking))

        deviations.sort(key=lambda d: abs(d[1] - d[2]), reverse=True)
        large_deviations_by_model[m.model_id] = deviations

    latex_lines = []
    latex_lines.append(r'\begin{tabular}{c ' + ' c ' * n + '}')
    latex_lines.append(r'\toprule')
    latex_lines.append(rf'\textbf{{Model}} & \multicolumn{{{n}}}{{c}}{{\textbf{{Top outlier skills}}}} \\')

    last_category = None
    models.sort(key=lambda m: overall_ranking.index(m.model_id))

    for model in models:
        if last_category != model.category:
            latex_lines.append(r'\midrule')
        last_category = model.category
        model_columns = []
        model_columns.append(model.display_name)

        for standard, original_rank, standard_rank in large_deviations_by_model[model.model_id][:n]:
            direction = '\\downto' if standard_rank > original_rank else '\\upto'
            model_columns.append(f'{standard} ({get_ordinal_str(original_rank)} {direction} {get_ordinal_str(standard_rank)})')

        latex_lines.append(' & '.join(model_columns) + r'\\')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')

    print('\n'.join(latex_lines))


def gsm8k_correlation(results: list[str]):
    results, standards, grades = load_results(results)

    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    models = []
    for row in table_config['models']:
        models.append(Model(*row))

    model_accuracy = {}
    model_name = {}

    with open('manual_data/gsm8k_accuracies.json', 'r') as f:
        gsm8k_accuracies_f = json.load(f)
        gsm8k_accuracies = gsm8k_accuracies_f['gsm8k_accuracies']

    data_points = []
    xy = []

    for model in models:
        model_name[model.model_id] = model.display_name
        model_accuracy[model.model_id] = compute_accuracy(results,
                                                          model.model_id,
                                                          lambda _, _p, _f: True)

        if model.model_id not in gsm8k_accuracies:
            continue

        data_points.append({
            "model": model.display_name,
            "accuracy": model_accuracy[model.model_id],
            "gsm8k_accuracy": gsm8k_accuracies[model.model_id]
        })
        xy.append((model_accuracy[model.model_id], gsm8k_accuracies[model.model_id]))

    print('Pearson Correlation between GSM8k and MathCAMPS:', pearsonr(*zip(*xy)))

    plot_vegalite('gsm_correlation', data_points, 'gsm8k_correlation')


def analyze_pythia_checkpoints(results: list[str]):
    all_results, standards, _ = load_results(results)

    models = []

    with open('eval-plans/hf-pythia.json') as f:
        eval_plan = json.load(f)
        models = eval_plan['model_names']
        model_id = models[0].split(':')[0]

    accuracies_by_standard = {}

    for standard in standards:
        accuracies_by_standard[standard] = {}
        for model in models:
            def standard_predicate(s, _p, _f):
                return s == standard
            accuracies_by_standard[standard][model] = \
                compute_accuracy(all_results, model, standard_predicate)

    last_model = models[-1]
    last_model_accuracies = {s: a[last_model] for s, a in accuracies_by_standard.items()}

    THRESHOLD = 0.3
    eval_standards = {s: a for s, a in last_model_accuracies.items() if a >= THRESHOLD}

    print(len(eval_standards), 'where the last checkpoint has accuracy >=', THRESHOLD)

    data_points = []

    for model in models:
        for standard in eval_standards:
            data_points.append({
                "training_steps": int(model.split(':step')[1]),
                "standard": standard,
                "accuracy": accuracies_by_standard[standard][model]
            })

    plot_vegalite('learning_dynamics', data_points, f'learning_dynamics')


def compare_two_models(results: list[str], m1: str, m2: str, k: int):
    all_results, standards, _ = load_results(results)

    standard_accuracies = {}
    for standard in standards:
        m1_accuracy = compute_accuracy(all_results, m1, lambda s, _p, _f: s == standard)
        m2_accuracy = compute_accuracy(all_results, m2, lambda s, _p, _f: s == standard)
        standard_accuracies[standard] = (m1_accuracy, m2_accuracy)

    m1_advantages = []
    m2_advantages = []
    for standard, (m1_acc, m2_acc) in standard_accuracies.items():
        delta = m1_acc - m2_acc
        if delta > 0:
            m1_advantages.append((standard, m1_acc, m2_acc, delta))
        elif delta < 0:
            m2_advantages.append((standard, m1_acc, m2_acc, delta))

    m1_advantages.sort(key=lambda x: x[3], reverse=True)
    m2_advantages.sort(key=lambda x: x[3])

    print(f"Top {k} standards where {m1} has the largest advantage over {m2}:")
    print("Standard\t\t{m1} Accuracy\t{m2} Accuracy\tDelta")
    for standard, m1_acc, m2_acc, delta in m1_advantages[:k]:
        print(f"{standard}\t\t{m1_acc:.2f}\t\t{m2_acc:.2f}\t\t{delta:.2f}")

    print(f"\nTop {k} standards where {m2} has the largest advantage over {m1}:")
    print("Standard\t\t{m2} Accuracy\t{m1} Accuracy\tDelta")
    for standard, m1_acc, m2_acc, delta in m2_advantages[:k]:
        print(f"{standard}\t\t{m2_acc:.2f}\t\t{m1_acc:.2f}\t\t{-delta:.2f}")


def pareto_comparison(results: list[str]):
    all_results, standards, _ = load_results(results)

    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    models = []
    for row in table_config['models']:
        models.append(Model(*row))

    model_accuracies = {}
    for model in models:
        model_accuracies[model.model_id] = {}
        for standard in standards:
            accuracy = compute_accuracy(all_results, model.model_id,
                                        lambda s, _p: s == standard)
            model_accuracies[model.model_id][standard] = accuracy

    pareto_better_count = 0
    total_pairs = 0

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]

            model1_pareto_better = True
            model2_pareto_better = True

            for standard in standards:
                if model_accuracies[model1.model_id][standard] < model_accuracies[model2.model_id][standard]:
                    model1_pareto_better = False
                if model_accuracies[model2.model_id][standard] < model_accuracies[model1.model_id][standard]:
                    model2_pareto_better = False

            if model1_pareto_better or model2_pareto_better:
                pareto_better_count += 1

            total_pairs += 1

    percentage = (pareto_better_count / total_pairs) * 100

    print(f"Pareto Comparison Results:")
    print(f"Number of model pairs where one model is Pareto-better: {pareto_better_count}")
    print(f"Total number of model pairs: {total_pairs}")
    print(f"Percentage of model pairs where one model is better-Pareto: {percentage:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-table', action='store_true',
                        help='Main LaTeX table of results.')
    parser.add_argument('--outlier-skills-table', action='store_true',
                        help='LaTeX table with outlier skills of each LLM.')
    parser.add_argument('--followups-accuracy-drop-table', action='store_true',
                        help='LaTeX table with accuracy drop with followups.')
    parser.add_argument('--pareto', action='store_true',
                        help='Compare pairs of models.')
    parser.add_argument('--results', type=str, nargs='*', help='Path to JSON files with results.')
    parser.add_argument('--gsm8k-correlation', action='store_true',
                        help='Correlation between GSM8k and MathCAMPS.')
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare two models and display top standards where each model has an advantage.')
    parser.add_argument('--pythia-checkpoints-eval', action='store_true',
                        help='Visualize CC learning dynamics using Pythia checkpoints.')
    parser.add_argument('--model1', type=str, help='Model ID of the first model to compare.')
    parser.add_argument('--model2', type=str, help='Model ID of the second model to compare.')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top standards to display for each model.')

    opt = parser.parse_args()

    if opt.main_table:
        generate_main_table(opt.results)
    elif opt.outlier_skills_table:
        generate_strengths_weaknesses_table(opt.results)
    elif opt.gsm8k_correlation:
        gsm8k_correlation(opt.results)
    elif opt.compare_models:
        compare_two_models(opt.results, opt.model1, opt.model2, opt.top_k)
    elif opt.pareto:
        pareto_comparison(opt.results)
    elif opt.followups_accuracy_drop_table:
        generate_followups_accuracy_drop_table(opt.results)
    elif opt.pythia_checkpoints_eval:
        analyze_pythia_checkpoints(opt.results)


if __name__ == '__main__':
    main()
