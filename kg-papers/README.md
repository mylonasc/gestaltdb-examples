## About
This is an example of using GestaltDB as a basis for agentic GraphRAG and creation of neuro-symbolic embeddings. 
In this repo, we are gathering a research corpus of papers around a particular topic (i.e., KG embeddings), 
we discoevr and analyze the ideas, methodologies, and results presented in them, create a property graph around them 

e.g.:

```
  [      Paper 123     ] -- [       Cites       ] -> [Paper 321]
  [      Paper 123     ] -- [ introduces_method ] -> [Method 345]
  [     Method 345     ] -- [    consists_of    ] -> [MethodFeatures XYZ]
  [ MethodFeatures XYZ ] -- [ described_by_tuple ] SamplingMethod XX, LearningAlgorithm YY, ML Architecture)
  
```
on a particular subject, parsing them, understanding the connections between them (using an agent), assigning them 
