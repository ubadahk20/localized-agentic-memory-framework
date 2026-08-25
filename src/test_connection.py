import ollama

response = ollama.chat(model="qwen2.5:1.5b", messages=[
                       {'role': 'user', 'content': 'Hello!'}])
print(response['message']['content'])
