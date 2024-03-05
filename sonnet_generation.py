import numpy as np

def sample_sonnet(hmmClass, snClass, seed = None):

    if seed is None:
        seed = np.random.randint(0,255)

    # save original seed
    seed0 = seed

    # Sample and convert sentence.
    sonnetList = []
    failBool = False
    for ldx in range(14):

        # get line
        thisLineList, sylList, seed = sample_sonnet_line(hmmClass, snClass, seed)

        # ensure that line has the right number of syllables
        lineSylCountListMax, lineSylCountListMin = countSyllables(snClass,thisLineList)
        cMax = sum(lineSylCountListMax)
        cMin = sum(lineSylCountListMin)
        if cMax == 10 or cMin == 10:
            thisLine = " ".join(thisLineList)
            thisLine = thisLine[0].capitalize() + thisLine[1:]
            sonnetList.append(thisLine)
        else:
            lineSylCountListMax, lineSylCountListMin = countSyllables(snClass,thisLineList)
            failBool = True
            print("Line has wrong number of syllables")
            
    if not failBool:
        for ldx in range(len(sonnetList)):
            print(sonnetList[ldx])
            
    return sonnetList, seed0

def sample_sonnet_line(hmmClass, snClass, seed):
    
    lineBool = False
    while lineBool == False:
        # get sample emission
        emission, states = hmmClass.generate_emission(15,seed)
        thisLine = [snClass.obs_map_r[i] for i in emission]

        # count syllables
        lineSylCountListMin = []
        lineSylCountListMax = []
        wordList = []
        for wdx in range(len(thisLine)):

            # check if we have enough syllables
            if sum(lineSylCountListMin) > 10:
                lineBool = False
                break
            elif sum(lineSylCountListMin) == 10:
                lineBool = True
                lineSylCountList = lineSylCountListMin.copy()
                break
            elif sum(lineSylCountListMax) == 10:
                lineBool = True
                lineSylCountList = lineSylCountListMax.copy()
                break
            word = thisLine[wdx]
            count = snClass.sylDict[word]
            if word == "i":
                wordList.append("I")
            else:
                wordList.append(word)

            # check for E
            eBool = False
            countInt = []
            for jdx in range(len(count)):
                if count[jdx][0] == 'E':
                    eBool = True
                    eCount = int(count[jdx][1])
                else:
                    countInt.append(int(count[jdx]))
            
            # get syllable count
            if eBool:
                if sum(lineSylCountListMax) + eCount == 10:
                    lineSylCountListMax.append(eCount)
                    lineSylCountListMin.append(eCount)
                    lineSylCountList = lineSylCountListMax.copy()
                    
                elif sum(lineSylCountListMin) + eCount == 10:
                    lineSylCountListMax.append(eCount)
                    lineSylCountListMin.append(eCount)
                    lineSylCountList = lineSylCountListMin.copy()
                else:
                    lineSylCountListMax.append(max(countInt))
                    lineSylCountListMin.append(min(countInt))
                    if sum(lineSylCountListMin) == 10:
                        lineSylCountListMin.append(100)
                    if sum(lineSylCountListMax) == 10:
                        lineSylCountListMax.append(100)
            else:
                lineSylCountListMax.append(max(countInt))
                lineSylCountListMin.append(min(countInt))
        seed += 1
    
    if len(wordList) != len(lineSylCountList):
        print("Word list and syllable counts have different lengths")

    return wordList, lineSylCountList, seed

def countSyllables(snClass,wordList):
    # count syllables
    lineSylCountListMin = []
    lineSylCountListMax = []
    for wdx in range(len(wordList)):

        # get word and count
        word = wordList[wdx].lower()
        count = snClass.sylDict[word]
        
        # check for E
        eBool = False
        countInt = []
        for jdx in range(len(count)):
            if count[jdx][0] == 'E':
                eBool = True
                eCount = int(count[jdx][1])
            else:
                countInt.append(int(count[jdx]))
        
        # get syllable count
        if wdx == len(wordList) - 1 and eBool:
            lineSylCountListMax.append(eCount)
            lineSylCountListMin.append(eCount)
        else:
            lineSylCountListMax.append(max(countInt))
            lineSylCountListMin.append(min(countInt))
    
    return lineSylCountListMax, lineSylCountListMin

