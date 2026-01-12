import argparse
import itertools
from pathlib import Path
import pandas as pd
import csv
import unicodedata
import time
import psutil
import threading


__version__ = "0.9.0"
DEFAULT_INPUT_FILE = Path("assets/Liste_Actions.csv")


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists in df among candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None

def normalize_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col)
    col = "".join(c for c in col if not unicodedata.combining(c))
    return col

def to_float(value) -> float | None:
    """Convert numbers written with comma, euro sign, spaces to float."""
    if pd.isna(value):
        return None
    text = str(value).strip().replace("€", "").replace(" ", "")
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_profit_percent(value) -> float | None:
    """
    Parse profit expressed as a percentage and return it as a float percent.
    Examples:
    - "5%" -> 5.0
    - "5"  -> 5.0
    - 0.2  -> 0.2  (interpreted as 0.2%)
    """
    if pd.isna(value):
        return None
    text = str(value).strip().replace(" ", "")
    text = text.replace("%", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def compute_benefit(cost_euros: float, profit_percent: float) -> float:
    """Compute final value after 2 years from cost and profit percent."""
    return cost_euros * (1 + profit_percent / 100)


def load_actions_from_excel(file_path: Path) -> list[dict]:
    """
    Load actions from an Excel or CSV file
    Return a list of action dictionaries.

    Expected columns:
    - "Actions #"
    - "Coût par action (en euros)"
    - "Bénéfice (après 2 ans)"

    :param file_path: Path to the Excel or CSV file
    :return: List of actions
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = file_path.suffix.lower()
    
    # Automatically choose the right reader based on file extension
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".ods":
        # Requires: pip install odfpy
        df = pd.read_excel(file_path, engine="odf")
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    
    # Normalize column names to make detection easier
    
    df.columns = [normalize_column_name(c) for c in df.columns]
    
    # ---- Column detection (supports both datasets) ----
    possible_name_cols = ["Actions #", "actions #", "name", "Actions", "action", "Action"]
    possible_cost_cols = [
        "Coût par action (en euros)", "cost", "price", "prix", "Coût", "cout"
        ]
    possible_profit_cols = [
        "Bénéfice (après 2 ans)", "profit", "bénéfice", "benefice"
                            ]
    name_col = pick_column(df, possible_name_cols)
    cost_col = pick_column(df, possible_cost_cols)
    profit_col = pick_column(df, possible_profit_cols)
    # Fallback: if detection failed, use column positions
    if name_col is None or cost_col is None or profit_col is None:
        if len(df.columns) >= 3:
            print("Warning: unable to detect columns by name, using default first 3 columns by position.")
            name_col = df.columns[0]
            cost_col = df.columns[1]
            profit_col = df.columns[2]
        else:
            raise ValueError(
                "Unsupported file format: unable to detect columns (need at least 3 columns).\n"
                f"Detected columns: {list(df.columns)}"
            )
    actions: list[dict] = []
    for _, row in df.iterrows():
        raw_name = row.get(name_col)
        if pd.isna(raw_name):
            continue
        name = str(raw_name).strip()
        if not name or name.lower() == "nan":
            continue

        cost = to_float(row.get(cost_col))
        profit_percent = parse_profit_percent(row.get(profit_col))

        # Filter invalid rows (dataset2 may contain negative prices/profits)
        if cost is None or profit_percent is None:
            continue
        if cost <= 0 or profit_percent <= 0:
            continue

        benefit = compute_benefit(cost, profit_percent)

        actions.append({
            "name": name,
            "cost": float(cost),
            "profit": float(profit_percent),   # stored as percent
            "benefit": float(benefit)
        })

    return actions


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def algo_bruteforce(actions, budget=500,):
    best_solution = []
    total_best_profit = 0.0
    best_total_cost = 0.0
    for size in range(1, len(actions) + 1):
        # Generate all combinations of the given size
        for combination in itertools.combinations(actions, size):

            # Calculate the total cost of the current combination
            total_cost = sum(action["cost"] for action in combination)

            # Check if the combination respects the budget constraint
            if total_cost <= budget:
                
                # Calculate the total profit in euros
                total_profit = sum(action["benefit"] - action["cost"]
                                   for action in combination)
                # If this combination is more profitable, keep it
                if total_profit > total_best_profit:
                    total_best_profit = total_profit
                    best_solution = combination
                    best_total_cost = total_cost

    # Return the most profitable valid combination
    return best_solution, best_total_cost, total_best_profit

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Action loader and processor"
    )
    parser.add_argument(
        "input_file",
        help="Path to the Excel or CSV file containing actions"
    )
    return parser.parse_args()



def main(input_file: Path = DEFAULT_INPUT_FILE) -> None:
    """
    Main entry point of the program.

    :param input_file: Path to the Excel or CSV file
    """
    print(f"Program version: {__version__}")
    print(f"Loading file: {input_file}")

    process = psutil.Process()

    # ---- Measure end-to-end (load + algo) ----
    start_e2e = time.perf_counter()
    mem_before_e2e = process.memory_info().rss

    actions = load_actions_from_excel(Path(input_file))

    # ---- Measure algo only ----
    mem_before_algo = process.memory_info().rss
    start_algo = time.perf_counter()

    best_solution, best_total_cost, total_best_profit = algo_bruteforce(actions)

    end_algo = time.perf_counter()
    mem_after_algo = process.memory_info().rss

    end_e2e = time.perf_counter()
    mem_after_e2e = process.memory_info().rss

    # Times
    elapsed_algo = end_algo - start_algo
    elapsed_e2e = end_e2e - start_e2e

    # Peak RSS (simple: max(before, after))
    peak_algo_mb = bytes_to_mb(max(mem_before_algo, mem_after_algo))
    peak_e2e_mb = bytes_to_mb(max(mem_before_e2e, mem_after_e2e))
    rss_after_run_mb = bytes_to_mb(mem_after_e2e)

    print("The best solution is:")
    print(best_solution)
    print(f"Total cost: {best_total_cost}")
    print(f"Total profit: {total_best_profit}")

    print(f"Execution time (algo only): {elapsed_algo:.6f} seconds")
    print(f"Execution time (load + algo): {elapsed_e2e:.6f} seconds")

    # Keep same “shape” of output, but report peak instead of delta
    print(f"Peak RSS (algo only): {peak_algo_mb:.2f} MB")
    print(f"Peak RSS (load + algo): {peak_e2e_mb:.2f} MB")
    print(f"RSS after run: {rss_after_run_mb:.2f} MB")
    print(f"Peak RSS (load + algo): {peak_e2e_mb:.2f} MB")
    print(f"RSS after run: {rss_after_run_mb:.2f} MB")


if __name__ == "__main__":
    args = parse_args()
    main(input_file=args.input_file)