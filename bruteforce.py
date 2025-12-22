import argparse
import itertools
from pathlib import Path
import pandas as pd


__version__ = "0.1.0"
DEFAULT_INPUT_FILE = Path("assets/Liste_Actions.csv")



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

    # Automatically choose the right reader based on file extension
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    actions = []

    for _, row in df.iterrows():
        name = row["Actions #"]
        cost = float(row["Coût par action (en euros)"])

        # Remove % sign and convert to decimal
        profit = float(str(row["Bénéfice (après 2 ans)"]).replace("%", "")) / 100 + 1

        benefit = cost * (profit)

        actions.append({
            "name": name,
            "cost": cost,
            "profit": profit,
            "benefit": benefit
        })

    return actions

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

    actions = load_actions_from_excel(Path(input_file))
    best_solution, best_total_cost, total_best_profit = algo_bruteforce(actions)
    print("The best solution is:")
    print(best_solution)
    print(f"Total cost: {best_total_cost}")
    print(f"Total profit: {total_best_profit}")

if __name__ == "__main__":
    args = parse_args()
    main(input_file=args.input_file)