from util.ModelUtil import get_llm

model = get_llm()

for chunk in model.stream("Why do parrots have colorful feathers?"):
    # 如果chunk.text不为空
    if chunk.text:
        print(chunk.text, end="|", flush=True)