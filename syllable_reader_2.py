def syllable_read(file):
    syllable = open(file, "r")
    syl = {}

    for x in syllable:
        w = x.split(" ")
        name = (w[0]).replace("\n", "")
        tmp = ["" for _ in range(2)]
        for j in range(1, len(w)):
            if len(w) > 2:
                if w[1][0] == "E":
                    tmp[0] = (w[2]).replace("\n", "")
                    tmp[1] = (w[1]).replace("E", "")
                elif w[2][0] == "E":
                    tmp[0] = (w[1]).replace("\n", "").replace("E", "")
                    tmp[1] = w[2]
                else:
                    tmp[0] = (w[1]).replace("E", "")
                    tmp[1] = (w[2]).replace("\n", "")
            else:
                tmp[0] = (w[1]).replace("\n", "")
                tmp[1] = (w[1]).replace("\n", "")

            syl[name] = tmp

    syllable.close()

    return syl
