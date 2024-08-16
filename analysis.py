#!/usr/bin/env python3


import ast
import collections
import re
from dataclasses import dataclass
from typing import Union, Tuple
import re
import ast
import argparse
import json

from scipy.stats import pearsonr
from tqdm import tqdm

import grammar
from grammar import load_standards


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

    def __eq__(self, other):
        return isinstance(other, Model) and self.model_id == other.model_id and self.vendor == other.vendor

    def __hash__(self):
        return hash((self.vendor, self.model_id))


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
                     followups=None,
                     return_responses=False) -> Union[Tuple[float, list], float]:
    total, correct = 0, 0
    responses = []

    for result in results.values():
        if result['model'] == model_id and predicate(result['standard'], result['problem'], None):
            total += 1
            responses.append(result)

            if result['correct']:
                if not followups:
                    correct += 1
                else:
                    total_followups, total_followups_correct = 0, 0

                    for followup in result['followups'].values():
                        if not predicate(result['standard'], result['problem'], followup['problem']):
                            continue

                        total_followups += 1
                        responses.append({**followup, 'is_followup': True})

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
        return (0, []) if return_responses else 0
    accuracy = correct / total
    return (accuracy, responses) if return_responses else accuracy


def fup_type_from_id(s):
    pattern = r'.*-([0-9]+)$'
    match = re.search(pattern, s)
    if match:
        num = match.group(1)
    else:
        num = None
    if num == '1':
        fup_type = 'ifup'
    elif num == '2':
        fup_type = 'cfup'
    else:
        raise ValueError("Non-followup ID passed")
    return fup_type


def compute_fup_accuracy(results: dict, model_id: str,
                         predicate,
                         followup_type) -> float:
    total_followups, total_followups_correct = 0, 0
    for result in results.values():
        if result['model'] == model_id and predicate(result['standard'], result['problem'], None):
            if result['correct']:
                for id, followup in result['followups'].items():
                    if not predicate(result['standard'], result['problem'], followup['problem']):
                        continue
                    type_fup = ast.literal_eval(id)

                    if fup_type_from_id(type_fup[0]) == followup_type:
                        total_followups += 1
                        if followup['correct']:
                            total_followups_correct += 1

    if total_followups == 0:
        return 0, 0
    return (total_followups_correct / total_followups), total_followups


def is_standard_from_grade(standard, grade):
    standard_grade = standard.split('.')[0]
    return standard_grade == grade


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


def generate_accuracy_table_data(results: list[str]):
    all_results, standards, grades = load_results(results)
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
            model_results.append({"accuracy": compute_accuracy(all_results,
                                                               model.model_id,
                                                               predicate),
                                  "grade": grade})
        results_table.append({
            'vendor': model.vendor,
            'model': model.display_name,
            'category': model.category,
            'model_id': model.model_id,
            'display_name': model.display_name,
            'is_open': model.category == 'open',
            'accuracies': model_results
        })

    results_table.sort(key=lambda x: (x['accuracies'][0]['accuracy']), reverse=True)
    return columns, results_table


def generate_standardwise_accuracy(results: list[str]):
    # model: {standard:accuracy}
    all_results, standards, grades = load_results(results)
    # Sort grades in ascending order, with 'K' coming first.
    grades = sorted(list(grades), key=lambda x: (x[0] != 'K', x))

    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    results_by_model = {}
    models_by_id = {}
    headers = ['Standard', 'Overall Acc.', 'IFUP Acc.', 'CFUP Acc.', 'Total FUPs']

    for row in table_config['models']:
        m = Model(*row)
        results_by_model[m.model_id] = {}
        models_by_id[m.model_id] = m

    followups_with_correct_answers, problems_with_followups, standards_with_followups = \
        get_followups_with_correct_answers(all_results)

    results_table = {}

    for i, model_id in tqdm(list(enumerate(results_by_model))):
        model = models_by_id[model_id]
        model_responses = []

        for grade in grades:
            results_by_model[model_id][grade] = []
            for standard in standards:
                standard_results = {}
                if is_standard_from_grade(standard, grade):
                    def standard_predicate(s, p, f):
                        return (s == standard and p in problems_with_followups)

                    # append main results
                    overall_acc = compute_accuracy(all_results,
                                                   model.model_id,
                                                   lambda s, _p, _f: s == standard,
                                                   return_responses=False)

                    _, responses = compute_accuracy(all_results,
                                                    model.model_id,
                                                    lambda s, _p, _f: s == standard,
                                                    followups='any',
                                                    return_responses=True)

                    model_responses.extend(responses)

                    standard_results['overall_acc'] = f"{overall_acc:.2f}"

                    # append ifup results
                    ifup_results, ifup_total = compute_fup_accuracy(all_results,
                                                                    model.model_id,
                                                                    standard_predicate,
                                                                    'ifup')

                    standard_results['num_ifup_corr'] = f"{ifup_results:.2f}" if ifup_results != 0 else "-"

                    # append cfup results
                    cfup_results, cfup_total = compute_fup_accuracy(all_results,
                                                                    model.model_id,
                                                                    standard_predicate,
                                                                    'cfup')
                    standard_results['num_cfup_corr'] = f"{cfup_results:.2f}" if cfup_results != 0 else "-"

                    # append total number of fups seen
                    total_fup_seen = ifup_total + cfup_total
                    standard_results['total_fup_seen'] = total_fup_seen if total_fup_seen != 0 else "-"

                    results_by_model[model_id][grade].append({'standard': standard, 'standard_results': standard_results})

        results_table[model_id] = {
            'vendor': model.vendor,
            'model': model.display_name,
            'category': model.category,
            'model_id': model.model_id,
            'display_name': model.display_name,
            'is_open': model.category == 'open',
            'accuracies': results_by_model[model_id],
            'responses': model_responses,
        }

    # results_table.sort(key=lambda x: (x['accuracies'][0]['accuracy']), reverse=True)
    return headers, results_table


def generate_main_table(results, html=False):
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

    table_contents.sort(key=lambda x: (x[0].category == "closed", x[1]), reverse=True)

    # Calculate and print grade-wise averages
    num_models = len(models)
    num_grades = len(columns)
    grade_sums = [0] * num_grades

    for model_results in results_table:
        for i, result in enumerate(model_results):
            grade_sums[i] += result

    print("Grade-wise averages:")
    for i, (grade, _) in enumerate(columns):
        average = grade_sums[i] / num_models
        print(f'{grade}: {average:.2f}')

    headers = ['Vendor', 'Model'] + [column[0] for column in columns]

    latex_lines = []
    html_lines = []

    latex_lines.append(r'\begin{tabular}{c c|' + ' '.join(['c'] * len(columns)) + '}')
    latex_lines.append(r'\toprule')
    latex_lines.append(' & '.join(['\\textbf{' + header + '}' for header in headers]) + r'\\')

    html_lines.append('<table>')
    html_lines.append('    <thead>')
    html_lines.append('        <tr>')
    for header in headers:
        html_lines.append(f'            <th>{header}</th>')
    html_lines.append('        </tr>')
    html_lines.append('    </thead>')
    html_lines.append('    <tbody>')

    last_category = None
    for model, *model_results in table_contents:
        if model.category != last_category:
            latex_lines.append(r'\midrule')
        last_category = model.category
        latex_lines.append(' & '.join([model.vendor, model.display_name] +
                                      [f'{result:.2f}' for result in model_results]) + r'\\')
        html_lines.append('        <tr>')
        html_lines.append(f'            <td>{model.vendor}</td>')
        html_lines.append(f'            <td>{model.display_name}</td>')
        for result in model_results:
            html_lines.append(f'            <td>{result:.2f}</td>')
        html_lines.append('        </tr>')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    html_lines.append('    </tbody>')
    html_lines.append('</table>')

    if not html:
        print('\n'.join(latex_lines))
    else:
        print('\n'.join(html_lines))


def generate_standards_table():
    print('--------------------------------------------------------------------------------------------------------------------')
    standards = load_standards()
    headers = ['Standard ID', 'Description']
    
    
    for grade in ['K', '1', '2', '3', '4', '5', '6', '7', '8']:
        latex_lines = []
        latex_lines.append(r'\begin{table}')
        latex_lines.append(r'\centering')
        latex_lines.append(r'\begin{tabular}{|p{2cm}|p{10cm}|}')
        latex_lines.append(r'\hline')
        latex_lines.append(' & '.join(['\\textbf{' + header + '}' for header in headers]) + r'\\')
        latex_lines.append(r'\hline')
        
        for standard in standards:
            id = standards[standard].id
            description = standards[standard].description
            if id[0] == grade: # checks that the first char is the grade
                latex_lines.append(f'{id} & {description}' + r'\\')
                latex_lines.append(r'\hline')
    
        latex_lines.append(r'\end{tabular}')
        latex_lines.append(r'\caption{CC Standards for Grade '  + grade + '}')
        latex_lines.append(r'\label{tab:standards' + f'-{grade}' + '}')
        latex_lines.append(r'\end{table}')

        print('\n'.join(latex_lines))

def generate_fup_table(results):
    all_results, _standards, grades = load_results(results)
    print(len(all_results), 'results loaded')


    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    models = []

    for row in table_config['models']:
        models.append(Model(*row))

    columns = [('All', lambda standard, _p, _f: True)]

    results_table = []
    
    headers = ['Vendor', 'Model', 'Main Acc.', 'IFUP Acc.', 'CFUP Acc.', 'Total FUPs seen']

    latex_lines = []
    latex_lines.append(r'\begin{tabular}{c c|c c c c}')
    latex_lines.append(r'\toprule')
    latex_lines.append(' & '.join(['\\textbf{' + header + '}' for header in headers]) + r'\\')

    for model in models:
        model_results = []
        for grade, predicate in columns:
            # append main results
            model_results.append(compute_accuracy(all_results,
                                                  model.model_id,
                                                  predicate))
                                     
            # append ifup results
            ifup_results, ifup_total = compute_fup_accuracy(all_results,
                                                  model.model_id,
                                                  predicate, 'ifup')
            
            # append cfup results
            model_results.append(ifup_results)
            cfup_results, cfup_total = compute_fup_accuracy(all_results,
                                                  model.model_id,
                                                  predicate, 'cfup')
            model_results.append(cfup_results)
            
            # append total number of fups seen
            model_results.append(ifup_total + cfup_total)
        # results_table.append(model_results)
        model_results = [f"{result:.2f}" for result in model_results]

        latex_lines.append(' & '.join([model.vendor, model.display_name] +
                                      model_results) + r'\\')
    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')

    print('\n'.join(latex_lines))

def generate_followups_accuracy_drop_table(results: list[str], n=2):
    standards_by_id = grammar.load_standards()
    standards = standards_by_id.values()
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

    models.sort(key=lambda m: (m.category == "open", overall_ranking.index(m.model_id)))

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
    latex_lines.append(r'\begin{tabular}{c c ' + ' c ' * 2*n + '}')
    latex_lines.append(r'\toprule')
    latex_lines.append(rf'\textbf{{Model}} & \textbf{{Acc. with follow-ups}} & \multicolumn{{{2*n}}}{{c}}{{\textbf{{Largest accuracy drop w/ follow-ups}}}} \\')

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
            model_columns.append(f'{standard} - {standards_by_id[standard].short_description}')
            model_columns.append(f'{no_followups_accuracy:.2f} \\downto {all_followups_accuracy:.2f})')

        latex_lines.append(' & '.join(model_columns) + r'\\')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')

    print('\n'.join(latex_lines))


def generate_strengths_weaknesses_table(results: list[str], n=1):
    results, standards, grades = load_results(results)

    all_standards = grammar.load_standards()

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
    latex_lines.append(r'\begin{tabular}{c ' + ' c ' * (2*n) + '}')
    latex_lines.append(r'\toprule')
    latex_lines.append(rf'\textbf{{Model}} & \textbf{{Top outlier skill}} & \textbf{{Rank change}} \\')

    last_category = None
    models.sort(key=lambda m: (m.category == "open", overall_ranking.index(m.model_id)))

    for model in models:
        if last_category != model.category:
            latex_lines.append(r'\midrule')

        last_category = model.category
        model_columns = []
        model_columns.append(model.display_name)

        for standard, original_rank, standard_rank in large_deviations_by_model[model.model_id][:n]:
            direction = '\\downto' if standard_rank > original_rank else '\\upto'
            model_columns.append(f'{standard} - {all_standards[standard].short_description}')
            model_columns.append(f' ({get_ordinal_str(original_rank)} {direction} {get_ordinal_str(standard_rank)})')

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


def generate_fup_table(results):
    all_results, _standards, grades = load_results(results)
    print(len(all_results), 'results loaded')

    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    models = []

    for row in table_config['models']:
        models.append(Model(*row))

    columns = [('All', lambda standard, _p, _f: True)]
    headers = ['Vendor', 'Model', 'Main Acc.', 'IFUP Acc.', 'CFUP Acc.',
               'Total FUPs seen']
    latex_lines = []
    latex_lines.append(r'\begin{tabular}{c c|c c c c}')
    latex_lines.append(r'\toprule')
    latex_lines.append(' & '.join(['\\textbf{' + header + '}'
                                   for header in headers]) + r'\\')
    latex_lines.append(r'\midrule')
    for model in models:
        model_results = []
        for grade, predicate in columns:
            # append main results
            model_results.append(compute_accuracy(all_results,
                                                  model.model_id,
                                                  predicate))
            # append ifup results
            ifup_results, ifup_total = compute_fup_accuracy(all_results,
                                                            model.model_id,
                                                            predicate, 'ifup')
            # append cfup results
            model_results.append(ifup_results)
            cfup_results, cfup_total = compute_fup_accuracy(all_results,
                                                            model.model_id,
                                                            predicate, 'cfup')
            model_results.append(cfup_results)
            # append total number of fups seen
            model_results.append(ifup_total + cfup_total)
        # results_table.append(model_results)
        model_results = [f"{result:.2f}" for result in model_results[:-1]] + [str(model_results[-1])]
        latex_lines.append(' & '.join([model.vendor, model.display_name] +
                                      model_results) + r'\\')
    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    print('\n'.join(latex_lines))


def generate_performance_by_standards(results):
    full_info_standards = load_standards()
    all_results, standards, _ = load_results(results)
    standards = sorted(list(standards), key=lambda x: (x[0] != 'K', x))
    # print(len(all_results), 'results loaded')
    
    # columns = [('All', lambda standard, _p, _f: True)]
    
    with open(TABLE_CONFIG_PATH, 'r') as f:
        table_config = json.load(f)

    models = []
    for row in table_config['models']:
        models.append(Model(*row))
    

    """
    # prints the list of model options
    for model in models:
        print(r'<option value="' + model.model_id + r'">' + model.display_name + '</option>')
    """
    
    """
    # prints the list of standards and their descriptions
    
    for standard in full_info_standards:
        print(f'\"{standard}\": \"{full_info_standards[standard].short_description}\",')
    """
    model_accuracies = {}
    for model in models:
        model_accuracies[model.model_id] = {}
        for standard in standards:
            accuracy = compute_accuracy(all_results, model.model_id, lambda s, _p, _f: s == standard)
            model_accuracies[model.model_id][standard] = accuracy
            
    data_dict_lines = []
    for model in models:
        data_dict_lines.append(f'\'' + f'{model.model_id}' + f'\'' + ': '+ '{')
        data_dict_lines.append(f'    name: \'{model.display_name}\',')
        data_dict_lines.append(r'    skills: {')
        for standard in standards:
            data_dict_lines.append(f'        \"' + standard + f'\": ' + f'\"{model_accuracies[model.model_id][standard]:.2f}\",')
        data_dict_lines.append(r'    }')
        data_dict_lines.append(r'},')
            
    
    print('\n'.join(data_dict_lines))
    
    """

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

    # Calculate and print grade-wise averages
    num_models = len(models)
    num_grades = len(columns)
    grade_sums = [0] * num_grades
    
    """
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
    parser.add_argument('--fup-table', action='store_true',
                        help='Table comparing followup performance.')
    parser.add_argument('--model1', type=str, help='Model ID of the first model to compare.')
    parser.add_argument('--model2', type=str, help='Model ID of the second model to compare.')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top standards to display for each model.')
    parser.add_argument('--standards-table', action='store_true',
                        help='Appendix table listing standards and descriptions')
    parser.add_argument('--performance-by-standards', action='store_true')
    parser.add_argument('--fup-table', action='store_true',
                        help='Table comparing followup performance.')
    parser.add_argument('--html', action='store_true',
                        help='When this tag is specified, the methods return the HTML code for making a table. The default without this option is LaTeX code.')
    opt = parser.parse_args()

    if opt.main_table:
        generate_main_table(opt.results, opt.html)
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
    elif opt.fup_table:
        generate_fup_table(opt.results)
    elif opt.standards_table:
        generate_standards_table()
    elif opt.fup_table:
        generate_fup_table(opt.results)
    elif opt.performance_by_standards:
        generate_performance_by_standards(opt.results)

if __name__ == '__main__':
    main()
