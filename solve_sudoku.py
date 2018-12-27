import datetime
import logging
import csv
import sys
import os
import pandas as pd
from typing import Dict, List, Tuple

from sudoku_data import SudokuData


debug_level = logging._nameToLevel[os.environ.get("DEBUG_LEVEL")] if os.environ.get("DEBUG_LEVEL") else logging.DEBUG
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s [%(name)s] [%(process)d] %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(debug_level)


def load_sudoku_csv() -> Dict[str, Dict[str, str]]:
    df = pd.read_csv('sudoku.csv').set_index('id')
    df_dict = df.to_dict(orient="index")
    return df_dict


def save_results_into_csv(sudoku_results_list: List[Tuple[int, str, int, int, float]]):
    with open(os.path.join('performance_summary.csv'), "w", newline="") as output_file:
        writer = csv.writer(output_file)
        columns = ['id', 'level', 'need_solver', 'elimination_time', 'time_spent']
        writer.writerow(columns)
        for row in sudoku_results_list:
            writer.writerow(row)
    logger.info('performance summary has been saved into a csv.')


def main():
    sudoku_results_list = []

    sudoku_dict = load_sudoku_csv()
    for id, value in sudoku_dict.items():
        level = value['level']
        sudoku_str = value['sudoku']

        logger.info('Starting solving Sudoku #%s, %s level.', id, level)

        start_time = datetime.datetime.now()
        sudoku_class = SudokuData(sudoku_str)
        print('Sudoku to solve:')
        sudoku_class.print_sudoku(sudoku_str)
        sudoku_class.solve_sudoku()
        print('Sudoku solution:')
        sudoku_class.show_solved_sudoku()

        time_spent = (datetime.datetime.now() - start_time).total_seconds()

        logger.info('Total time: %s seconds', time_spent)

        sudoku_results_list.append((int(id), level, int(sudoku_class.need_solver), sudoku_class.elimination_time,
                                    time_spent))

        logger.info('Done with Sudoku #%s', id)

    save_results_into_csv(sudoku_results_list)


if __name__ == '__main__':
    main()
