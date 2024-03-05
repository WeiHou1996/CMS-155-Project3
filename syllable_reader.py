def len_syllable(file):
    syllable = open(file, "r")
    num_words = 0
    for y in syllable:
        num_words += 1
    syllable.close()

    return num_words


def syllable_read(file):
    num_words = len_syllable(file)
    syllable = open(file, "r")
    syl = [["" for _ in range(3)] for _ in range(num_words)]

    for i, x in enumerate(syllable):
        w = x.split(" ")
        syl[i][0] = (w[0]).replace("\n", "")
        for j in range(1, len(w)):
            if len(w) > 2:
                if w[1][0] == "E":
                    syl[i][1] = (w[2]).replace("\n", "")
                    syl[i][2] = (w[1]).replace("E", "")
                elif w[2][0] == "E":
                    syl[i][1] = (w[1]).replace("\n", "").replace("E", "")
                    syl[i][2] = w[2]
                else:
                    syl[i][1] = (w[1]).replace("E", "")
                    syl[i][2] = (w[2]).replace("\n", "")
            else:
                syl[i][1] = (w[1]).replace("\n", "")
                syl[i][2] = (w[1]).replace("\n", "")

    syllable.close()

    return syl
