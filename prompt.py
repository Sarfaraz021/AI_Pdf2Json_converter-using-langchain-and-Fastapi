template = """
INSTRUCTIONS: 

You are a task-specific domain agent. Analyze the content of the provided document and formulate a response in JSON format.


            Context:
            {context}

            User Query:
            {question}

            Provide your answer as a valid JSON object:
            AI: Let's think step by step:
            """
