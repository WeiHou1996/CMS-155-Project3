def syllable_read(file):
    syllable = open(file, "r")
    syl = {}

    for x in syllable:
        line = x.split(" ")
        word = (line[0]).replace("\n", "")
        count = []
        for j in range(1, len(line)):
            thisCount = line[j].replace("\n","")
            count.append(thisCount)
        
        syl[word] = count

    syllable.close()
    
    return syl
