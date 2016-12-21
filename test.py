import classify as c

# file = open("pos.txt", 'r', encoding="ISO-8859-1").read()
# sentences = [p for p in file.split('\n')]
#
# for sent in sentences:
#     print(c.sentiment(sent))


while True:
    sent = input("Text to Analyze (At least two coherent sentences) or 1 to exit :")
    if sent == "1":
        break
    print(sent, c.sentiment(sent))
