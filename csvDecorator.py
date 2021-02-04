import csv


class csvDecorate():
    def __init__(self, pat):
        self.path = pat

    def csv_decorator(func):
        def func_wrapper(self):
            reader = func(self)
            for row in reader:
                self.data[row[0]][row[1]] = row[2]
            return func_wrapper