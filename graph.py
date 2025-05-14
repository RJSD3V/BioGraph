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
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
import pprint
from langgraph.graph import StateGraph, START, END

#Manage Environment Variables. 
from dotenv import load_dotenv
import mermaid as md 
from mermaid.graph import Graph 
import os , getpass
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def _set_env(var:  str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")


load_dotenv()
auradb_conn=os.environ["AURADB_CONN"]
auradb_username = os.environ["AURADB_USERNAME"]
auradb_password = os.environ["AURADB_PASSWORD"]



class State(TypedDict):
    text:str
    nodes: Optional[List[dict]]
    is_graph_pushed: Optional[bool]
    messages: Annotated[list, add_messages]





@tool("push_graph")
def push_graph(state:State) -> Any:

    """ Push the json object with nodes and relationsips between entities to an AuraDB Graph database instance

    Args:
         state : State -> state object that holds the json property from the State that 
                            is essentially a list of  json objects or Python list of dicts that 
                            has the graph data with nodes and relationships.

    Returns:
          dict : Updates the state object by returning if the graph push was successful

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

    input_data = state["nodes"] # Your provided JSON here
    
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
        
            
            return{"messages":{"role":"assistant","content":f"Graph push successful with {summary['nodes'].counters} nodes and {summary['relationships'].counters}"}
                   , "is_graph_pushed": True
                   }
        
    except Exception as e:
        return{"messages":[{"role":"assistant","content":f"Graph push failed with error: {str(e)}"
            }],
            "is_graph_pushed": False
            }
    
@tool("process_passage")
def process_passage(state:State) -> Any:
    
    """ 
     Identifies node and relationships between entities in the input text passage. 
     The objective of the function is to return a json object or Python dictionary, which will be passed
     to the push_graph function that stores it in the auradb graph databse. 

     args:
        text: str -> State object which has the text property to be used to parse nodes and relationships from. 
      
     returns:
        dict: python dictionary that outlines nodes and relationships between entities in the given passage
     
    
    """
    prompt = PromptTemplate.from_template("""For the following passage, give me ONLY a serialized json object that 
    describes the relationship between the species, compatible with a graph data structure. 
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

    """)
    
    parser_llm = ChatOllama(model="mistral:instruct")
    formatted_prompt = prompt.format(text=state["text"])
    response = parser_llm.invoke(formatted_prompt)
    parser = JsonOutputParser()
    return {"messages":[{"role":"assistant","content":f"Graph Created Successfully", "body": parser.parse(response.content)}],"json": parser.parse(response.content)}
    


builder = StateGraph(State)
builder.add_node("process_passage", process_passage)
builder.add_node("push_graph",push_graph)
builder.add_edge(START, "process_passage")
builder.add_edge("process_passage","push_graph")
builder.add_edge("push_graph", END)


graph = builder.compile()
mermaid_code = graph.get_graph().draw_mermaid()
print(mermaid_code)
render = md.Mermaid(mermaid_code)
print(render)


passage = """
In the tropical rainforest ecosystem, several species exhibit a variety of biological relationships that illustrate evolutionary, ecological, and taxonomic connections. The Jaguar (Panthera onca) is a top predator and preys upon the Capybara (Hydrochoerus hydrochaeris) and the Green Anaconda (Eunectes murinus). Both the Capybara and Green Anaconda share the same wetland habitat, demonstrating ecological co-occurrence.
The Green Anaconda often competes with the Harpy Eagle (Harpia harpyja) for prey such as the Howler Monkey (Alouatta palliata). The Howler Monkey, an arboreal primate, is closely related to the Spider Monkey (Ateles geoffroyi), with both belonging to the family Atelidae. This taxonomic relationship indicates a common evolutionary ancestor.
The Harpy Eagle also preys on the Sloth (Bradypus variegatus), which has a mutualistic relationship with several species of algae, such as Trentepohlia spp., that grow on its fur, providing camouflage. Additionally, the Sloth shares a symbiotic relationship with the Moth (Cryptoses choloepi), which uses the sloth’s fur for habitat.
From an evolutionary standpoint, the Capybara and the Jaguar belong to different orders: Rodentia and Carnivora, respectively, reflecting deep phylogenetic divergence. The Sloth is taxonomically grouped within the order Pilosa, distinct from the other mammals mentioned.
Together, these species form a complex web of predation, competition, symbiosis, and shared evolutionary history, making the tropical rainforest a dynamic and interconnected ecosystem.
"""



messages = []
result = graph.invoke({"messages":messages, "text":passage})

print("Messages:", result)
