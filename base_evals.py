from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=nvidia/Nemotron-H-8B-Base-8K,trust_remote_code=True,load_in_4bit=True",
    tasks=["arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"],
    device="cuda",
    batch_size=1,
)
print(results)
