#!/usr/bin/env python3
"""Convert words_enwik8.dic to clean cmix format (one word per line, lowercase only).
Parses it the same way cmix does: extract runs of a-z."""

words = []
with open('dictionary/words_enwik8.dic', 'rb') as f:
    data = f.read()

word = ''
for b in data:
    c = chr(b) if b < 128 else ''
    if c >= 'a' and c <= 'z':
        word += c
    else:
        if word:
            words.append(word)
            word = ''
if word:
    words.append(word)

# Take first 44880 (max dictionary capacity)
words = words[:44880]
print(f"Extracted {len(words)} entries")

with open('dictionary/words_enwik8_parsed.dic', 'w') as f:
    for w in words:
        f.write(w + '\n')
print("Saved to dictionary/words_enwik8_parsed.dic")
