# # Basic Prompt
# from langchain_core.prompts import PromptTemplate

# template = """You are a helpful assistant.
# Question: {question}
# Context: {context}
# Answer:"""
# prompt = PromptTemplate(
#  input_variables=["question", "context"],
#  template=template,
# )
# print(prompt) 
# # input_variables=['question'] input_types={} partial_variables={} template='You are a helpful assistant.\nQuestion: {question}\nAnswer:'
# print(prompt.invoke({"question":"What is LangChain?", "context":'medical records'}))
# print(prompt.format(question="What is LangChain?", context= 'medical records'))



# from langchain.prompts import FewShotPromptTemplate, PromptTemplate, Example

# examples = [
#  Example(input="2+2", output="4"),
#  Example(input="3+5", output="8"),
# ]
# example_prompt = PromptTemplate(
#  template="Input: {input}\nOutput: {output}\n",
#  input_variables=["input", "output"],
# )
# few_shot = FewShotPromptTemplate(
#  examples=examples,
#  example_prompt=example_prompt,
#  prefix="Answer the following math questions.",
#  suffix="Question: {question}\nAnswer:",
#  input_variables=["question"],
#  example_separator="\n",
# )
# print(few_shot.format(question="7+6"))

