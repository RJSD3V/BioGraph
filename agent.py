from typing import Annotated
from typing import Any, Optional , List
from typing_extensions import TypedDict
import os, json, logging
from neo4j import GraphDatabase

#Langgraph and Langchain dependencies. 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
#from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
import ast
from langgraph.prebuilt import tools_condition, ToolNode
import os , getpass
from langgraph.graph import StateGraph , END
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def _set_env(var:  str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

#Manage Environment Variables. 
from dotenv import load_dotenv

load_dotenv()
auradb_conn=os.environ["AURADB_CONN"]
auradb_username = os.environ["AURADB_USERNAME"]
auradb_password = os.environ["AURADB_PASSWORD"]


from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: Annotated[list, add_messages]



@tool("push_graph")
def push_graph(graph: str | list) -> str:

    """  
    Push the JSON object with nodes and relationships to an AuraDB Graph database instance.
    
    Args:
        graph (str): A Serialized JSON object as a string containing a list of dicts created by the process_passage tool.
    Returns:
        str: A serialized string of the summary of objects successfully pushed to Auradb


    """

    AURA_CONNECTION_URI = auradb_conn
    AURA_USERNAME = "neo4j"
    AURA_PASSWORD = auradb_password

    # Driver instantiation
    driver = GraphDatabase.driver(
        AURA_CONNECTION_URI,
        auth=(AURA_USERNAME, AURA_PASSWORD)
    )

    import uuid
    
    if isinstance(graph, list): 
        input_data = graph
    elif isinstance(graph, str):
        obj= ast.literal_eval(graph)
        loaded = json.loads(obj)
        print(loaded)
        print(type(loaded))
        input_data = loaded["graph"] if "graph" in loaded else loaded
    else:
        raise ValueError("Invalid type for graph argument")
    
    print(type(input_data))
    print(input_data)

    # Transform nodes
    nodes = {item["node"]: {"id": str(uuid.uuid4())} for item in input_data}

    # Transform relationships
    relationships = []
    for item in input_data:
        source_node = item["node"]
        for rel in item["relationships"]:
            relationships.append({
                "source": source_node,
                "target": rel.get("target", rel.get("source")),
                "type": rel["type"].upper().replace(" ", "_")
            })


    def create_graph(tx):
        # Create nodes with unique IDs
        for name, props in nodes.items():
            nodes_result = tx.run("""
                MERGE (n:Species {name: $name})
                SET n.id = $id
            """, name=name, id=props["id"])

        # Create relationships
        for rel in relationships:
            relationships_result = tx.run("""
                MATCH (a:Species {name: $source}), (b:Species {name: $target})
                MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
            """, source=rel["source"], target=rel["target"], type=rel["type"])
    
        return {"nodes": nodes_result.consume(), "relationships": relationships_result.consume()}

    
    try:
        with driver.session() as session:
            summary = session.execute_write(create_graph)
        
            return True
        
    except Exception as e:
        return False
    

@tool("process_passage")
def process_passage(text:str) -> dict:
    
    """
    Identifies node and relationships between entities in the input text passage.
    Args:
        text (str): The passage to parse and retrieve nodes and relationships from.
    Returns:
        str: A Serialized JSON string consisting a list of nodes and relationships this 
            is to be structured by push_graph tool and uploaded to the graph database.
    """


    prompt = PromptTemplate.from_template("""For the following passage, output ONLY a complete, valid serialized JSON array of node objects as described below.
        Do not add any explanation or comments. Output must parse with json.loads().  
    For example a graph relationship: Lion --> (predator) --> Antelope should output a serialized json string \n

    the passage starts from the next line \n {text} \n


    *** Requirements ***

        - Only return the json object. Nothing else.
        - each node relationship should have only one source or target. Not a list. 
        - Add any node and relationship properties (ecosystem, etc.) based on the passage

    *** Example output for a relationship ***

    {{
       "node": "Jaguar",
       "relationships": [
         {{
           "type": "predator",
           "target": "Capybara"
         }}
       ]
     }}
                                          
    Stick to the schema given.

    """)
    
    parser_llm = ChatOllama(model="mistral:instruct", max_tokens=8192)
    json_parser = JsonOutputParser()
# Wrap it with OutputFixingParser
    fixing_parser = OutputFixingParser.from_llm(parser=json_parser, llm=parser_llm)

    output_schema = { "type": "array",
                      "items": { 
                  "type": "object",
                  "properties": 
                      { "node": { "type": "string" },
                            "relationships": { "type": "array",
                                  "items": { "type": "object",
                                          "properties": { "type": { "type": "string" },
                                                  "target": { "type": "string" } },
                                                  "required": ["type", "target"] } 
                                                },
                                                "ecosystem": { "type": "string" } 
                                                },
                                                "required": ["node", "relationships", "ecosystem"] } }
    
    structured_llm = parser_llm.with_structured_output(output_schema)

    formatted_prompt = prompt.format(text=text)
    response = structured_llm.invoke(formatted_prompt)
    
    
    try:
       
       parsed = fixing_parser.parse(response)
       print("Parsed JSON",parsed)
       return json.dumps({"graph":parsed})
    except Exception as e:
        
        return json.dumps({"messages":"Graph creation failed. Try again."})


llm = ChatOllama(model="mistral")
tools = [process_passage, push_graph]
llm_with_tools = llm.bind_tools(tools)

from langchain_core.runnables import RunnableConfig

def agent(
        state:State, 
        config:RunnableConfig
        ):
    system_prompt = SystemMessage(
        """You are a helpful research assistant who needs to answer questions about ecosystems using the available tools. 
            For each step, call the appropriate tool and wait for its result before proceeding to the next.
            Do not answer directly or do any reasoning yourself.
            Always respect the required order of operations when using tools."""
    )
    response = llm_with_tools.invoke([system_prompt] + state["messages"], config)
    print(response)
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))   
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition)

workflow.add_edge("tools","agent")

agent = workflow.compile()
