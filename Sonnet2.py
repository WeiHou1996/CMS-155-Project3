def Sonnet(file):
    Sonnet = []
    Sonnets = []

    f = open("shakespeare.txt", "r")
    c = 1
    cc = 0
    for x in f:
        w = x.split(" ")
        if w[-1] == f"{c}\n":
            c += 1
            if cc == 14:
                Sonnets.append(Sonnet)
            Sonnet = []
            cc = 0
            continue
        if w[0] == "\n":
            continue
        Sonnet.append(w)
        cc += 1

    f.close()

    words = []

    c = 0
    for Sonnet in Sonnets:
        for line in Sonnet:
            for w in line:
                w = (
                    w.replace(",", "")
                    .replace("\n", "")
                    .replace(":", "")
                    .replace(";", "")
                    .replace(".", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("!", "")
                    .replace("?", "")
                )
                if w != "":
                    if (w[-1] == "'") or (w[0] == "'"):
                        w = w.replace("'", "")
                    words.append(w.lower())

    words = set(words)

    obs_map = {}
    obs_map_r = {}

    for i, w in enumerate(words):
        obs_map[w] = i
        obs_map_r[i] = w

    lines = [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 13]]
    # lines = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]

    rhyme_lines = []
    obs_lines = []

    for Sonnet in Sonnets:
        for j in lines:  # get list of 2 lines needed
            line = []
            obs = []
            for k in j:  # go through those 2 lines
                for l in reversed(Sonnet[k]):  # go word by word
                    if l == "":
                        continue
                    w = (
                        l.replace(",", "")
                        .replace(":", "")
                        .replace("\n", "")
                        .replace(";", "")
                        .replace(".", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("!", "")
                        .replace("?", "")
                    ).lower()
                    if w != "":
                        if (w[-1] == "'") or (w[0] == "'"):
                            w = w.replace("'", "")
                        line.append(w)
                        obs.append(obs_map[w])
            rhyme_lines.append(line)
            obs_lines.append(obs)

    # for Sonnet in Sonnets:
    #     for j in range(14):  # get list of 2 lines needed
    #         line = []
    #         obs = []
    #         for l in reversed(Sonnet[j]):  # go word by word
    #             if l == "":
    #                 continue
    #             w = (
    #                 l.replace(",", "")
    #                 .replace(":", "")
    #                 .replace("\n", "")
    #                 .replace(";", "")
    #                 .replace(".", "")
    #                 .replace("(", "")
    #                 .replace(")", "")
    #                 .replace("!", "")
    #                 .replace("?", "")
    #             ).lower()
    #             if w != "":
    #                 if (w[-1] == "'") or (w[0] == "'"):
    #                     w = w.replace("'", "")
    #                 line.append(w)
    #                 obs.append(obs_map[w])
    #     rhyme_lines.append(line)
    #     obs_lines.append(obs)

    return obs_lines, obs_map, obs_map_r, words
