An attempt to create a 'chatbot' that will have a very short conversation: it will ask if a user wants to listen to some music and if so, it will ask to describe situation they are in (e.g. 'Going to the beach').

It will then analyse the situation by assigning it valency (pos, neg, neutral) and one of the brown corpus categories (adventure, romance, etc.).

The next step is to find a song where lyrics have the same valency, category. There will be more than one - so will be narrowing down by cosine similarity between the situation and the text in the lyrics.
