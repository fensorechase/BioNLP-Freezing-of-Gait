from nltk.corpus import wordnet

syns = wordnet.synsets('shoot')
for syn in syns:
    print('Synset name:', syn.name())
    print('Synset meaning:', syn.definition())
    print('Examples:', syn.examples())
    print('Synonyms:', syn.lemmas())
    for lemma in syn.lemmas():
        print (lemma.name())
    print('---')