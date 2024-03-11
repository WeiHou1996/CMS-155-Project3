from nltk.corpus import cmudict

# nltk.download("cmudict")


def rhyme_endings(words):
    word_dict = cmudict.dict()
    ss = {}

    for word in words:
        try:
            tmp = word_dict[word.lower()]
            mat = []
            for i in range(len(tmp)):
                w = tmp[i]
                for j in reversed(range(len(w))):
                    if w[j][-1] == "1":
                        mat.append(w[j:])
                        break

            ss[word] = mat
        except:  # noqa: E722
            ss[word] = []
            pass

    return ss


def find_rhymes(word, ss):
    tmp = []
    end = ss[word]
    for e in end:
        for key in ss.keys():
            if e in ss[key]:
                if key == word:
                    continue
                tmp.append(key)

    return set(tmp)
