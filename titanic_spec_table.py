import json
import pandas as pd
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

# Read table
DATA = pd.read_csv('data/titanic.csv')

# Read metadata
with open('data/metadata.json') as f:
    METADATA = json.load(f)

repl = PythonREPL()
repl.locals = {'df': DATA.sample(5)}
repl_tool = Tool(
    name="python_repl",
    description="A pandas dataframe instance. Use this to execute pandas.DataFrame commands. Input should be a valid pandas query. Remember to always `import pandas` prior to your queries.",
    func=repl.run,
)

SYSTEM_MESSAGE = f"""
# PERSONA
You are a help museum assistant about the Titanic victims. Take your user message and query your tool to get best answer.

# TOOL INFORMATION
- You tool provides information about the Titanic victims, to query it you MUST translate the user message to a pandas.DataFrame query.
- Remember to always `import pandas` before providing your query

## EXAMPLE
Example 1:
<user> how many people were in Titanic?
<assistant> import pandas; df.shape[0]
<assistant> There were ..... people in Titanic

Example 2:
<user> how many people survived Titanic?
<assistant> import pandas; df[df['Survived']==1].count()
<assistant> There were .... who survived

# RESPONSE FORMAT
- Always give friendly responses.
"""

prompt = ChatPromptTemplate(
    [
        ('system', SYSTEM_MESSAGE),
        ('placeholder', '{messages}'),
    ]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
)

# React agent using pandas.DF as a research tool 
agent = create_react_agent(
    model=llm,
    prompt=prompt,
    tools=[repl_tool],
    debug=True,
)

if __name__ == '__main__':
    
    query = input('Q: ')
    
    while query != 'quit':
        response = agent.invoke(
            input={
                "messages": [('user', query)]
            }
        )
        print('R:', response['messages'][-1].content)
        print()
        query = input('>')