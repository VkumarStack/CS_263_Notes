import datasets
import evaluate

def get_mafund_samples(num_samples=10):
    """
    Loads the Mafand en-amh test dataset and extracts the first 'num_samples' examples.
    """
    # Load the specific 'en-amh' configuration
    dataset = datasets.load_dataset("masakhane/mafand", "en-amh", split="test")
    
    amharic_texts = []
    english_refs = []
    
    for i in range(num_samples):
        item = dataset[i]
        pair = item['translation']
        
        # 'tgt' is Amharic (Source input for this task), 'src' is English (Reference)
        if 'tgt' in pair:
            am_text = pair['tgt']
            en_text = pair['src']
        else:
            # Fallback for safe key retrieval
            am_text = pair.get('amh', pair.get('am'))
            en_text = pair.get('en')
            
        amharic_texts.append(am_text)
        english_refs.append(en_text)
        
    return amharic_texts, english_refs

def compute_avg_sacrebleu(predictions, references):
    """
    Computes the average SacreBLEU score.
    """
    metric = evaluate.load("sacrebleu")
    formatted_refs = [[ref] for ref in references]
    results = metric.compute(predictions=predictions, references=formatted_refs)
    return results["score"]

def print_translation_pairs(inputs, predictions):
    """
    Prints the Input and Prediction pairs in the requested format.
    """
    print("--- Translation Pairs ---")
    # zip combines the two lists so we can iterate over them simultaneously
    for i, (source, pred) in enumerate(zip(inputs, predictions)):
        print(f"Pair {i+1}")
        print(source)
        print(pred)
        print() # Adds an empty line between pairs for readability

# --- Main Execution ---

# 1. Get the data
amharic_inputs, english_references = get_mafund_samples(10)

# 2. Print prompts for you to copy to ChatGPT
print("--- COPY TO CHATGPT ---")
for i, am_text in enumerate(amharic_inputs):
    print(f"[{i+1}] {am_text}")
print("\n" + "="*30 + "\n")

# 3. (Placeholders) After you get answers from ChatGPT, put them in this list:
# Replace the empty strings below with the actual ChatGPT outputs.
chatgpt_predictions = [
  "Yesterday, Jordan amended its Press and Publications Law to restrict the right to freedom of expression on the Internet.",
  "At the same time, journalists and free expression advocates staged a peaceful protest in front of the parliament where the law was being passed.",
  "Advocates of free expression, holding banners bearing the slogan “Internet freedom,” have warned of a trend toward the death of the Internet in Jordan.",
  "For this decree dubbed the “Internet funeral,” the protesters were dressed in black.",
  "The approved law is still awaiting the signature of King Abdullah II, which—along with the approval of the upper house of parliament—is required before it can come into force and before all laws are implemented in Jordan.",
  "Jamil Nimeri, a member of parliament who opposed the law, went beyond opposing it and also took part in the protest.",
  "According to their statement, such proclamations serve only to silence the voices of the people by restricting freedom.",
  "Blog owners take responsibility even for the comments that their readers post on their blogs.",
  "Amid strong opposition to the draft bill, an online campaign was immediately launched to draw attention to the uproar it would spark among netizens.",
  "On Twitter, netizens expressed their frustration in writing."
]

# 4. Check if results are filled, then print formatted pairs and score
if any(chatgpt_predictions) and len(chatgpt_predictions) == len(amharic_inputs):
    
    # Print the formatted pairs as requested
    print_translation_pairs(amharic_inputs, chatgpt_predictions)
    
    # Compute and print the score
    score = compute_avg_sacrebleu(chatgpt_predictions, english_references)
    print("-" * 30)
    print(f"Average SacreBLEU Score: {score:.2f}")
    
elif not any(chatgpt_predictions):
    print("Please paste the ChatGPT translations into the 'chatgpt_predictions' list in the code.")
else:
    print("Error: The number of predictions does not match the number of inputs (10).")