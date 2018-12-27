import datetime
from collections import OrderedDict
import logging
import sys
import os
import math
from itertools import product
from typing import Dict, List, Set, Union

from constraint import *

debug_level = logging._nameToLevel[os.environ.get("DEBUG_LEVEL")] if os.environ.get("DEBUG_LEVEL") else logging.DEBUG
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s [%(name)s] [%(process)d] %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(debug_level)


def is_int(value) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


class CellData(object):
    def __init__(self, cell_row_name: str, cell_column_name: str, cell_init_value: str):
        self._cell_row_name = cell_row_name  # type: str
        self._cell_column_name = cell_column_name  # type: str
        self._cell_name = cell_row_name + cell_column_name  # type: str
        self._cell_init_value = cell_init_value  # type: str
        self._cell_possible_solution = self._find_cell_init_solution(cell_init_value)  # type: List[Union[int, None]]
        self._square_name = None  # type: str

    def __repr__(self):
        return "cell '{}' has initial value '{}', with possible solution as {}".format(self._cell_name,
                                                                                       self._cell_init_value,
                                                                                       self.cell_possible_solution)

    @staticmethod
    def _find_cell_init_solution(cell_init_value: str) -> List[int]:
        if is_int(cell_init_value):
            return [int(cell_init_value)]
        else:
            return []

    @property
    def cell_row_name(self) -> str:
        return self._cell_row_name

    @property
    def cell_column_name(self) -> str:
        return self._cell_column_name

    @property
    def cell_name(self) -> str:
        return self._cell_name

    @property
    def cell_possible_solution(self) -> List[int]:
        return self._cell_possible_solution

    def update_cell_possible_solution(self, possible_solution: List[int]):
        self._cell_possible_solution = possible_solution

    @property
    def square_name(self) -> str:
        return self._square_name

    @square_name.setter
    def square_name(self, square_name: str):
        self._square_name = square_name

    def get_solution(self) -> int:
        if len(self.cell_possible_solution) == 1:
            return self.cell_possible_solution[0]
        else:
            raise Exception('There are more than one solution for the cell %s, thus unable to get the solution.',
                            self.cell_name)


class CellDataCollection(object):
    def __init__(self, collection_type: str, collection_name: str):
        # collection type can only be one of ['Row', 'Column', 'Square']
        self._collection_type = collection_type  # type: str
        self._collection_name = collection_name  # type: str
        self._cell_data_dict = OrderedDict()  # type: Dict[str, CellData]

    def __repr__(self):
        return "{} {} has the following cells: {}".\
            format(self._collection_type, self._collection_name,
                   [cell_name for cell_name, cell_data in sorted(self._cell_data_dict.items())])

    def add_cell_data(self, cell_data: CellData):
        self._cell_data_dict[cell_data.cell_name] = cell_data

    def get_all_cell_data_as_dict(self) -> Dict[str, CellData]:
        return self._cell_data_dict

    def get_all_cell_data_as_list(self) -> List[CellData]:
        return [cell_data for cell_name, cell_data in self._cell_data_dict.items()]

    def get_all_cell_names(self) -> List[str]:
        return [cell_name for cell_name, cell_data in sorted(self._cell_data_dict.items())]

    def get_one_cell_data(self, cell_name: str) -> CellData:
        if cell_name in self._cell_data_dict:
            return self._cell_data_dict[cell_name]
        else:
            raise Exception('{} is not in row {}'.format(cell_name, self._collection_name))


class SudokuData(object):
    global_index_list = [n for n in range(0, 9)]
    row_name_master_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    column_name_master_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    cell_name_master_list = sorted([row_name + col_name for (row_name, col_name)
                                    in product(row_name_master_list, column_name_master_list)])
    square_name_master_list = ['sqr_1', 'sqr_2', 'sqr_3', 'sqr_4', 'sqr_5', 'sqr_6', 'sqr_7', 'sqr_8', 'sqr_9']
    # square_column_name_list_mapping: for a given square name, mapping the list of column names
    square_column_name_list_mapping = {'sqr_1': ['1', '2', '3'],
                                       'sqr_2': ['4', '5', '6'],
                                       'sqr_3': ['7', '8', '9'],
                                       'sqr_4': ['1', '2', '3'],
                                       'sqr_5': ['4', '5', '6'],
                                       'sqr_6': ['7', '8', '9'],
                                       'sqr_7': ['1', '2', '3'],
                                       'sqr_8': ['4', '5', '6'],
                                       'sqr_9': ['7', '8', '9']
                                       }
    completed_possible_solution = [n for n in range(1, 10)]

    def __init__(self, sudoku_to_solve: str):
        self._sudoku_to_solve = self._check_sudoku_to_solve(sudoku_to_solve)  # type: str
        self._row_str_dict = self._split_into_row_str()  # type: Dict[str, str]
        self._all_cell_data_dict = self._prep_cell_data()  # type: Dict[str, CellData]
        self._number_of_solved_cells = self._summarize_total_number_of_solved_cells()  # type: int

        self._row_data_dict = self._group_row_data()  # type: Dict[str, CellDataCollection]
        self._column_data_dict = self._group_column_data()  # type: Dict[str, CellDataCollection]
        self._square_data_dict = self._group_square_data()  # type: Dict[str, CellDataCollection]

        self._row_existing_solution_dict = OrderedDict()  # type: Dict[str, Set[Union[int, None]]]
        self._column_existing_solution_dict = OrderedDict()  # type: Dict[str, Set[Union[int, None]]]
        self._square_existing_solution_dict = OrderedDict()  # type: Dict[str, Set[Union[int, None]]]

        self._elimination_time = 1  # type: int

        self._sudoku_solver = Problem()  # type: Problem
        self._sudoku_solution_str_list = []  # type: List[Union[str, None]]

        self._need_solver = False  # type: bool

    @property
    def need_solver(self) -> bool:
        return self._need_solver

    @property
    def elimination_time(self) -> int:
        return self._elimination_time

    @staticmethod
    def _check_sudoku_to_solve(sudoku_to_solve: str) -> str:
        if len(sudoku_to_solve) == 81:
            return sudoku_to_solve
        else:
            raise Exception('Invalid Sudoku to solve. {} out of 81 values were provided.'.
                            format(len(sudoku_to_solve)))

    @staticmethod
    def print_sudoku(sudoku_str: str):
        if len(sudoku_str) == 81:
            sudoku_to_print = []
            index = 0
            while index <= 80:
                str_section = sudoku_str[index: index+9]
                message = ''
                for str_index, value in enumerate(str_section):
                    message += value + ' '
                    if str_index in [2, 5]:
                        message += '| '
                sudoku_to_print.append(message)
                if index in [18, 45]:
                    sudoku_to_print.append('------+-------+------')

                index += 9

            for r in sudoku_to_print:
                print(r)
        else:
            raise Exception('Invalid Sudoku to print. {} out of 81 values were provided.'.
                            format(len(sudoku_to_solve)))

    def _split_into_row_str(self) -> Dict[str, str]:
        row_str_dict = dict()

        for index in self.global_index_list:
            row_name = self.row_name_master_list[index]
            str_section = self._sudoku_to_solve[9 * index: 9 * (index + 1)]
            row_str_dict[row_name] = str_section

        return row_str_dict

    def _prep_cell_data(self) -> Dict[str, CellData]:
        cell_data_dict = OrderedDict()
        for row_name, row_str in self._row_str_dict.items():
            for index, value in enumerate(row_str):
                col_name = self.column_name_master_list[index]
                cell_data = CellData(row_name, col_name, value)
                cell_data_dict[cell_data.cell_name] = cell_data

        return cell_data_dict

    def _summarize_total_number_of_solved_cells(self) -> int:
        total_number = 0
        for cell_name, cell_data in self._all_cell_data_dict.items():
            if len(cell_data.cell_possible_solution) == 1:
                total_number += 1

        return total_number

    def _group_row_data(self) -> Dict[str, CellDataCollection]:
        row_data_dict = OrderedDict()
        for index in self.global_index_list:
            row_name = self.row_name_master_list[index]
            row_data = CellDataCollection('Row', row_name)
            # going through self._cell_data_dict (Dict[str, CellData]) to add cell data into corresponding row
            for cell_name, cell_data in self._all_cell_data_dict.items():
                if cell_data.cell_row_name == row_name:
                    row_data.add_cell_data(cell_data)
            row_data_dict[row_name] = row_data

        return row_data_dict

    def _group_column_data(self) -> Dict[str, CellDataCollection]:
        column_data_dict = OrderedDict()
        for index in self.global_index_list:
            column_name = self.column_name_master_list[index]
            column_data = CellDataCollection('Column', column_name)
            # going through self._cell_data_dict (Dict[str, CellData]) to add cell data into corresponding column
            for cell_name, cell_data in self._all_cell_data_dict.items():
                if cell_data.cell_column_name == column_name:
                    column_data.add_cell_data(cell_data)
            column_data_dict[column_name] = column_data

        return column_data_dict

    def _group_square_data(self) -> Dict[str, CellDataCollection]:
        square_data_dict = OrderedDict()
        for index in self.global_index_list:
            square_name = self.square_name_master_list[index]
            square_data = CellDataCollection('Square', square_name)
            row_from_index = 3 * math.floor(index / 3)
            row_end_index = 3 * math.floor(index / 3 + 1)
            square_row_name_list = self.row_name_master_list[row_from_index: row_end_index]
            square_column_name_list = self.square_column_name_list_mapping[square_name]
            square_cell_name_list = [row_name + col_name for (row_name, col_name)
                                     in product(square_row_name_list, square_column_name_list)]
            # going through self._cell_data_dict (Dict[str, CellData]) to add cell data into corresponding square
            for cell_name, cell_data in self._all_cell_data_dict.items():
                if cell_name in square_cell_name_list:
                    square_data.add_cell_data(cell_data)
                    # set the square name for the cell data
                    self._all_cell_data_dict[cell_name].square_name = square_name

            square_data_dict[square_name] = square_data

        return square_data_dict

    @staticmethod
    def _find_existing_solution(cell_data_list: List[CellData]) -> Set[Union[int, None]]:
        existing_solution = set()
        for cell_data in cell_data_list:
            if len(cell_data.cell_possible_solution) == 1:
                existing_solution.add(cell_data.get_solution())

        return existing_solution

    def _find_all_existing_solution(self):
        # find existing solution for rows
        for row_name, row_cell_data_collection in self._row_data_dict.items():
            row_cell_data_list = row_cell_data_collection.get_all_cell_data_as_list()
            self._row_existing_solution_dict[row_name] = self._find_existing_solution(row_cell_data_list)
        # find existing solution for columns
        for column_name, column_cell_data_collection in self._column_data_dict.items():
            column_cell_data_list = column_cell_data_collection.get_all_cell_data_as_list()
            self._column_existing_solution_dict[column_name] = self._find_existing_solution(column_cell_data_list)
        # find existing solution for squares
        for square_name, square_cell_data_collection in self._square_data_dict.items():
            square_cell_data_list = square_cell_data_collection.get_all_cell_data_as_list()
            self._square_existing_solution_dict[square_name] = self._find_existing_solution(square_cell_data_list)

    def _eliminate_existing_solution(self):
        for cell_name, cell_data in self._all_cell_data_dict.items():
            if len(cell_data.cell_possible_solution) != 1:
                # eliminate row existing solution:
                row_name = cell_data.cell_row_name
                row_existing_solution = self._row_existing_solution_dict[row_name]
                # eliminate column existing solution:
                column_name = cell_data.cell_column_name
                column_existing_solution = self._column_existing_solution_dict[column_name]
                # eliminate square existing solution:
                square_name = cell_data.square_name
                square_existing_solution = self._square_existing_solution_dict[square_name]

                # combine existing solutions from row, column, and square
                all_existing_solution_set = row_existing_solution.union(column_existing_solution,
                                                                        square_existing_solution)
                filtered_possible_solution_list = [i for i in self.completed_possible_solution
                                                   if i not in list(all_existing_solution_set)]
                cell_data.update_cell_possible_solution(filtered_possible_solution_list)

    def _elimination(self):
        logger.debug('%s out of 81 cells have been populated with the solution. Starting the 1st elimination.',
                     self._number_of_solved_cells)
        self._find_all_existing_solution()
        self._eliminate_existing_solution()
        post_elimination_number = self._summarize_total_number_of_solved_cells()

        while post_elimination_number > self._number_of_solved_cells:
            self._elimination_time += 1
            if self._elimination_time == 2:
                logger_message = 'Number of cells solved increased from {} to {} after the 1st elimination. Now ' \
                                 'trying the 2nd eliminations.'.format(self._number_of_solved_cells,
                                                                       post_elimination_number)
            elif self._elimination_time == 3:
                logger_message = 'Number of cells solved increased from {} to {} after the 2nd elimination. Now ' \
                                 'trying the 3rd eliminations.'.format(self._number_of_solved_cells,
                                                                       post_elimination_number)
            elif self._elimination_time == 4:
                logger_message = 'Number of cells solved increased from {} to {} after the 3rd elimination. Now ' \
                                 'trying the 4th eliminations.'.format(self._number_of_solved_cells,
                                                                       post_elimination_number)
            else:
                logger_message = 'Number of cells solved increased from {} to {} after the {}th elimination. Now ' \
                                 'trying the {}th eliminations.'.format(self._number_of_solved_cells,
                                                                        post_elimination_number,
                                                                        self._elimination_time - 1,
                                                                        self._elimination_time)
            logger.debug(logger_message)

            self._number_of_solved_cells = post_elimination_number
            self._find_all_existing_solution()
            self._eliminate_existing_solution()
            post_elimination_number = self._summarize_total_number_of_solved_cells()

    def _add_constraint(self, cell_data_collection: CellDataCollection):
        cell_name_list = cell_data_collection.get_all_cell_names()
        self._sudoku_solver.addConstraint(AllDifferentConstraint(), cell_name_list)

    def _call_csp_solver(self):
        # add all the 81 variables
        for cell_name, cell_data in self._all_cell_data_dict.items():
            possible_solution_list = cell_data.cell_possible_solution
            self._sudoku_solver.addVariable(cell_name, possible_solution_list)

        # add constraint for each row:
        for row_name, row_cell_data_collection in self._row_data_dict.items():
            self._add_constraint(row_cell_data_collection)
        # add constraint for each column:
        for column_name, column_cell_data_collection in self._column_data_dict.items():
            self._add_constraint(column_cell_data_collection)
        # add constraint for each square:
        for square_name, square_cell_data_collection in self._square_data_dict.items():
            self._add_constraint(square_cell_data_collection)

        # get solution
        solution_dict_list = self._sudoku_solver.getSolutions()

        # format solution:
        solution_str_list = []
        for solution_dict in solution_dict_list:
            solution_list = [solution_dict[cell_name] for cell_name in sorted(solution_dict)]
            solution_str = ''
            for sol in solution_list:
                solution_str += str(sol)
            solution_str_list.append(solution_str)

        # save the solutions as strings in a list into the class
        self._sudoku_solution_str_list = solution_str_list

    def solve_sudoku(self):
        self._elimination()

        if self._number_of_solved_cells == 81:
            solved_sudoku_str = ''
            # construct solution into a string
            for cell_name in self.cell_name_master_list:
                solved_sudoku_str += str(self._all_cell_data_dict[cell_name].get_solution())
            self._sudoku_solution_str_list = [solved_sudoku_str]
        else:
            logger.info('%s out of 81 cells have been found the solution after eliminations. Starting CSP solver.',
                        self._number_of_solved_cells)
            self._need_solver = True
            self._call_csp_solver()

    def show_solved_sudoku(self):
        if len(self._sudoku_solution_str_list) == 0:
            raise Exception('No solution to show')
        elif len(self._sudoku_solution_str_list) == 1:
            self.print_sudoku(self._sudoku_solution_str_list[0])
        else:
            logger.info('there are %s solutions.', len(self._sudoku_solution_str_list))
            for index, sudoku_solution_str in enumerate(self._sudoku_solution_str_list):
                logger.info('Solution #%s:', index)
                self.print_sudoku(sudoku_solution_str)


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('sudoku.csv').set_index('id')
    df_dict = df.to_dict(orient="index")

    id = int(sys.argv[1])

    sudoku_to_solve = df_dict[id]['sudoku']

    start_time = datetime.datetime.now()

    sudoku = SudokuData(sudoku_to_solve)

    sudoku.print_sudoku(sudoku_to_solve)

    sudoku.solve_sudoku()

    sudoku.show_solved_sudoku()

    logger.info('Total time: %s seconds', (datetime.datetime.now() - start_time).total_seconds())
