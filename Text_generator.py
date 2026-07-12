# ==========================================
# AI Internship Task-4
# Generative Text Model using GPT-2
# ==========================================

from transformers import pipeline

print("=" * 60)
print("        GENERATIVE TEXT MODEL USING GPT-2")
print("=" * 60)

generator = pipeline(
    "text-generation",
    model="gpt2"
)

while True:
    prompt = input("\nEnter a topic (or type 'exit' to quit): ")

    if prompt.lower() == "exit":
        print("\nThank you for using the Text Generator!")
        break

    print("\nGenerating text...\n")

    result = generator(
        prompt,
        max_length=200,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        truncation=True
    )

    print("=" * 60)
    print(result[0]["generated_text"])
    print("=" * 60)
