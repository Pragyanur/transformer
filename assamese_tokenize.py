import nltk
import pickle as pkl

# with open('../datasets/assamese2vec_trial.pkl', 'r',encoding='utf-8') as file:
#   words = pkl.load(file)
#   print(words[:100])

with open("asm_wikipedia_2021_100K-sentences.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    # Remove newline character from each line (optional)
    sentences = [line.strip() for line in lines]
    # dictionary of index:string
    sentence_idx = {idx: line.strip() for idx, line in enumerate(lines)}


for key in range(5):
    print(sentence_idx[key])
    print("\n")

# sentences = [line.split("\t")[1] for idx, line in enumerate(sentences)]


assamese_text = " ".join(sentences)

with open("assamese_text.txt", "w", encoding="utf-8") as file:
    file.write(assamese_text)

# Assuming you have some Assamese text data in a variable called 'assamese_text'
assamese_words = assamese_text.split(" ")

# remove full-stop from words
count_full_stop = 0
for word in assamese_words:
    if word[-1] == "।":
        word = word[: len(word) - 1]
        count_full_stop += 1

print(count_full_stop)

unique_assamese_words = list(set(assamese_words))

# with open('assamese_words.txt', 'w', encoding='utf-8') as file:
#   file.write("".join(unique_assamese_words))

print(len(assamese_words))
print(unique_assamese_words)
# print(unique_assamese_words[:500])


# print(assamese_words[:10])
