def syllable_read(file):
    syllable = open(file, "r")
    syl = {}

    for x in syllable:
        w = x.split(" ")
        name = (w[0]).strip()
        name = name.replace("\n", "")
        if (name[0] == "'") or (name[-1] == "'"):
            name = name.replace("'", "")
        tmp = ["" for _ in range(2)]
        for j in range(1, len(w)):
            if len(w) > 2:
                if "E" in w[1]:
                    tmp[0] = (w[2]).replace("\n", "").replace("E", "")
                    tmp[1] = (w[1]).replace("\n", "").replace("E", "")
                elif "E" in w[2]:
                    tmp[0] = (w[1]).replace("\n", "").replace("E", "")
                    tmp[1] = w[2]
                else:
                    tmp[0] = (w[1]).replace("\n", "").replace("E", "")
                    tmp[1] = (w[2]).replace("\n", "").replace("E", "")
            else:
                tmp[0] = (w[1]).replace("\n", "").replace("E", "")
                tmp[1] = (w[1]).replace("\n", "").replace("E", "")

            syl[name] = tmp

    syllable.close()

    return syl
