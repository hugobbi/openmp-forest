"""
plot_results.py
---------------
Lê os arquivos results/*.txt gerados pelo dt_program e plota:
  - Tempo de build por número de threads (para cada estratégia)
  - Speedup por número de threads
  - Eficiência por número de threads

Uso:
    python3 plot_results.py [--results-dir ./results] [--output-dir ./plots]

Formato esperado dos arquivos .txt:
    build time=<float>
    test accuracy=<float> (X/Y)
    avg inference time=<float>
    ...
    threads=<int>
    parallelizaiton strategy=<char>   # (typo original do código)
"""

import os
import re
import argparse
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configurações de estilo ────────────────────────────────────────────────────
STRATEGY_LABELS = {
    's': 'Sequencial (s)',
    'p': 'Splits sempre (p)',
    't': 'Splits c/ threshold (t)',
    'n': 'Só nós (n)',
    'P': 'Splits+nós sempre (P)',
}
STRATEGY_COLORS = {
    's': '#6c757d',
    'p': '#0077b6',
    't': '#f77f00',
    'n': '#d62828',
    'P': '#2dc653',
}
# Estratégias que fazem sentido variar threads (excluir 's' dos gráficos de speedup/eficiência)
PARALLEL_STRATEGIES = ['p', 't', 'n', 'P']

plt.rcParams.update({
    'font.family': 'monospace',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})


# ── Parser dos arquivos de resultado ──────────────────────────────────────────
def parse_result_file(path):
    """Retorna dict com os campos do arquivo de resultado, ou None se inválido."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, _, val = line.partition('=')
                data[key.strip()] = val.strip()

    try:
        result = {
            'build_time': float(data['build time']),
            'threads': int(data['threads']),
            # typo original do código: "parallelizaiton"
            'strategy': data.get('parallelizaiton strategy',
                                  data.get('parallelization strategy', '?')).strip(),
        }
    except (KeyError, ValueError):
        return None
    return result


def load_results(results_dir):
    """
    Retorna estrutura:
        { strategy: { threads: [build_time, ...] } }
    """
    data = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(results_dir, fname)
        r = parse_result_file(path)
        if r is None:
            print(f'  [aviso] arquivo ignorado (formato inesperado): {fname}')
            continue
        data[r['strategy']][r['threads']].append(r['build_time'])
    return data


def aggregate(data):
    """
    Calcula média e desvio padrão por (strategy, threads).
    Retorna: { strategy: { threads: (mean, stdev) } }
    """
    agg = {}
    for strategy, thread_dict in data.items():
        agg[strategy] = {}
        for threads, times in thread_dict.items():
            mean = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0.0
            agg[strategy][threads] = (mean, std)
    return agg


# ── Funções de plot ────────────────────────────────────────────────────────────
def plot_build_time(agg, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, thread_dict in sorted(agg.items()):
        if not thread_dict:
            continue
        xs = sorted(thread_dict)
        ys = [thread_dict[t][0] for t in xs]
        errs = [thread_dict[t][1] for t in xs]
        label = STRATEGY_LABELS.get(strategy, strategy)
        color = STRATEGY_COLORS.get(strategy, None)
        ax.errorbar(xs, ys, yerr=errs, marker='o', label=label,
                    color=color, capsize=4, linewidth=2)

    ax.set_xlabel('Número de threads')
    ax.set_ylabel('Tempo de build (s)')
    ax.set_title('Tempo de construção da árvore de decisão')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    out = os.path.join(output_dir, 'build_time.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Salvo: {out}')


def plot_speedup(agg, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Linha ideal
    all_threads = sorted({t for td in agg.values() for t in td})
    ax.plot(all_threads, all_threads, '--', color='#aaa', linewidth=1.2, label='Speedup ideal')

    for strategy in PARALLEL_STRATEGIES:
        thread_dict = agg.get(strategy)
        if not thread_dict:
            continue
        xs = sorted(thread_dict)
        if 1 not in thread_dict:
            print(f'  [aviso] estratégia "{strategy}" sem medição com 1 thread — speedup ignorado')
            continue
        t1 = thread_dict[1][0]
        speedups = [t1 / thread_dict[t][0] for t in xs]
        label = STRATEGY_LABELS.get(strategy, strategy)
        color = STRATEGY_COLORS.get(strategy, None)
        ax.plot(xs, speedups, marker='o', label=label, color=color, linewidth=2)

    ax.set_xlabel('Número de threads')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup — árvore de decisão com OpenMP')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    out = os.path.join(output_dir, 'speedup.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Salvo: {out}')


def plot_efficiency(agg, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    all_threads = sorted({t for td in agg.values() for t in td})
    ax.axhline(1.0, color='#aaa', linestyle='--', linewidth=1.2, label='Eficiência ideal')

    for strategy in PARALLEL_STRATEGIES:
        thread_dict = agg.get(strategy)
        if not thread_dict:
            continue
        xs = sorted(thread_dict)
        if 1 not in thread_dict:
            continue
        t1 = thread_dict[1][0]
        efficiencies = [(t1 / thread_dict[t][0]) / t for t in xs]
        label = STRATEGY_LABELS.get(strategy, strategy)
        color = STRATEGY_COLORS.get(strategy, None)
        ax.plot(xs, efficiencies, marker='o', label=label, color=color, linewidth=2)

    ax.set_xlabel('Número de threads')
    ax.set_ylabel('Eficiência  (speedup / n_threads)')
    ax.set_title('Eficiência — árvore de decisão com OpenMP')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    out = os.path.join(output_dir, 'efficiency.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Salvo: {out}')


def plot_strategy_comparison(agg, output_dir):
    """Barras: tempo de build por estratégia fixando max threads."""
    all_threads = sorted({t for td in agg.values() for t in td})
    max_threads = max(all_threads)

    strategies = [s for s, td in agg.items() if max_threads in td]
    if not strategies:
        print('  [aviso] nenhuma estratégia com threads máximo — comparação ignorada')
        return

    times = [agg[s][max_threads][0] for s in strategies]
    errs  = [agg[s][max_threads][1] for s in strategies]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    colors = [STRATEGY_COLORS.get(s, '#888') for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, times, yerr=errs, color=colors, capsize=5, width=0.5)
    ax.set_ylabel('Tempo de build (s)')
    ax.set_title(f'Comparação de estratégias ({max_threads} threads)')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    out = os.path.join(output_dir, 'strategy_comparison.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Salvo: {out}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Gera gráficos de desempenho do dt_program')
    parser.add_argument('--results-dir', default='./results',
                        help='Pasta com os arquivos .txt de resultado (default: ./results)')
    parser.add_argument('--output-dir', default='./plots',
                        help='Pasta onde os gráficos serão salvos (default: ./plots)')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f'Erro: pasta de resultados não encontrada: {args.results_dir}')
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Lendo resultados de: {args.results_dir}')
    data = load_results(args.results_dir)

    if not data:
        print('Nenhum resultado encontrado. Verifique o formato dos arquivos .txt.')
        return

    agg = aggregate(data)

    print(f'\nEstatísticas encontradas:')
    for strategy in sorted(agg):
        for threads in sorted(agg[strategy]):
            mean, std = agg[strategy][threads]
            label = STRATEGY_LABELS.get(strategy, strategy)
            print(f'  [{label}] {threads:2d} threads: {mean:.4f}s ± {std:.4f}s')

    print(f'\nGerando gráficos em: {args.output_dir}')
    plot_build_time(agg, args.output_dir)
    plot_speedup(agg, args.output_dir)
    plot_efficiency(agg, args.output_dir)
    plot_strategy_comparison(agg, args.output_dir)

    print('\nPronto!')


if __name__ == '__main__':
    main()
