template = """
INSTRUCTIONS: 

You are a task-specific domain agent. Analyze the content of the provided document and formulate a response in JSON format.
You will convert user's data into porper json format.
            Context:
            {context}

            User Query:
            {question}

            Provide your answer as a valid JSON object:
            AI: Let's do it step by step:
            """
