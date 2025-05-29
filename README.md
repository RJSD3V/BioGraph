# BioGraph - Using AI to build Complex Knowledge and Action Graph from Dense Information


## Overview

This project is developed solely as an exercise to understand to what end can a large language model understand really dense and comprehensive information very well. 
To be more precise, can a large language model, given the right prompts and instructions, draw out meaningful entities and relationshiips between concepts and their interconnectedness , 
in complex subjects like Biology, Medicine and Law? If yes, can they help professional visualise and recommend suggestions if prompted? It's like putting a stress test on top of 
different language models to see how far can they think and if not provide a concrete innovative solution, maybe help lay out data representations that can help humans decide next steps, 
in a complicated situation. 


## Langgraph as a scaffold for intelligence

Im using Langgraph here as I get to experiment with several control flows with different nodes constructed for different purposes. This is quite basic at the moment with two main nodes for parsing and pushing the knowledge graph to the graph database, but it will also include a vector database that stores said corpus indexed properly so that only appropriate content relevant to the query is 
extracted and structured for the parser. 


## To Get started

### Keys

Make sure you have several keys handy, including: 

- The AuraDB instance key and url that you need to access AuraDB. Im using a free instance.
- OpenAI key, in case you're using a tool calling to turn the langgraph into an agent, although if its a deterministic graph that can use on premise models using ollama or huggingface Id discourage wasting money to build an agent where a simple graph with a human in the loop is sufficient.
- Im not using one at the moment, but in case you're using a cloud based VectorStore like Pinecone, you would want to store a key for that too. Although i personally would recommend a docker version of pgvector.


## Start with the notebooks. 

 You can get started with the notebook called `simple_graph`. Which is a straightforward end to end graph you can visualize and run from start to finish (Provided you get your keys in order)
 Progressing on to the agentic part is not really necessary, although you can if you want to just to play around (Just make sure you ahve enough Openai Credits)






