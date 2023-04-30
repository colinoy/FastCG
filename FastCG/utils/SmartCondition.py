import logging
class SmartCondition():
    condition_to_lambda = {
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        "in": lambda x, y: x in y,
        "not in": lambda x, y: x not in y,
        "is": lambda x, y: x is y,
        "is not": lambda x, y: x is not y,
        "equal": lambda x, y: x == y,
        "not equal": lambda x, y: x != y,
        "greater": lambda x, y: x > y,
        "greater or equal": lambda x, y: x >= y,
        "less": lambda x, y: x < y,
        "less or equal": lambda x, y: x <= y,
    }



    def __init__(self, target, condition):
        logging.info(f"Initializing SmartCondition with target: {target} and condition: {condition}")
        self.target = target
        if condition and condition in self.condition_to_lambda:
            self.condition = self.condition_to_lambda[condition]
        else:
            logging.critical("Condition is not in the list of supported conditions")
            raise ValueError("Condition is not defined")

        
    def __call__(self, data,target=None):
        if target:
            return self.condition(data, target)
        return self.condition(data, self.target)
    
    def check(self, data, target=None):
        if target:
            return self.condition(data, target)
        return self.condition(data, self.target)
    
