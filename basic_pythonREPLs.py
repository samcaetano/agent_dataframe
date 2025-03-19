# Content from: https://api.python.langchain.com/en/latest/utilities/langchain_experimental.utilities.python.PythonREPL.html
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL # type: ignore
from langchain_experimental.tools import PythonREPLTool

class SimplePythonShell():
    """
    Define a simple Python Read-Eval-Print Loop
    """
    def __init__(self):
        self.python_repl = PythonREPL() 

    def run(self, query : str):
        return self.python_repl.run(query)
    
class MyPythonShell():
    """
    Define a simple Python Read-Eval-Print Loop
    """
    def __init__(self):
        self.python_repl = PythonREPL(locals={'x': 102030})

    def run(self, query : str):
        return self.python_repl.run(query)

if __name__ == '__main__':
    
    repl = PythonREPL()
    response = repl.run("print('hello world')")
    print(response)

    # Access variable
    repl.locals = {'x': 10} # pass external variable
    response = repl.run("print(x+x)")
    print(response)
    
    # Loop
    response = repl.run("while True: print(x)", timeout=3) # Timeout to solve: 'timeout=?'
    print(response)
