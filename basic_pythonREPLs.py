# Content from: https://api.python.langchain.com/en/latest/utilities/langchain_experimental.utilities.python.PythonREPL.html
from langchain_experimental.utilities import PythonREPL # type: ignore

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
