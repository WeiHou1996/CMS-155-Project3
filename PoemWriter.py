import numpy as np

class PoemWriter:
    def __init__(self,snClass,hmmClass):
        self.hmmClass = hmmClass
        self.snClass = snClass
        self.poemStructure = []

    def write_poem(self,seed=None):

        # create random number generator
        rng = np.random.default_rng(seed=seed)

        # Sample and convert sentence.
        poemList = []
        failBool = False
        for ldx in range(len(self.poemStructure)):
            
            # get number of syllables
            cTarget = self.poemStructure[ldx]

            # get line
            thisLineList, sylList = self.write_line(rng,cTarget)

            # ensure that line has the right number of syllables
            lineSylCountListMax, lineSylCountListMin = self.countSyllables(thisLineList)
            cMax = sum(lineSylCountListMax)
            cMin = sum(lineSylCountListMin)
            if cMax == cTarget or cMin == cTarget:
                thisLine = " ".join(thisLineList)
                thisLine = thisLine[0].capitalize() + thisLine[1:]
                if ldx == len(self.poemStructure)-1:
                    thisLine += "."
                poemList.append(thisLine)
            else:
                lineSylCountListMax, lineSylCountListMin = self.countSyllables(thisLineList)
                failBool = True
                raise Exception("Line has wrong number of syllables")
                
        if not failBool:
            for ldx in range(len(poemList)):
                print(poemList[ldx])
                
        return poemList

    def write_line(self,rng,cTarget):
        
        # get classes
        hmmClass = self.hmmClass
        snClass = self.snClass

        # iterate until appropriate line is written
        lineBool = False
        while lineBool == False:
            # get sample emission
            emission, states = hmmClass.generate_emission(cTarget,rng)
            thisLine = [snClass.obs_map_r[i] for i in emission]

            # count syllables
            lineSylCountListMin = []
            lineSylCountListMax = []
            wordList = []
            for wdx in range(len(thisLine)):

                # check if we have enough syllables
                if sum(lineSylCountListMin) > cTarget:
                    lineBool = False
                    break
                elif sum(lineSylCountListMin) == cTarget:
                    lineBool = True
                    lineSylCountList = lineSylCountListMin.copy()
                    break
                elif sum(lineSylCountListMax) == cTarget:
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
                    if sum(lineSylCountListMax) + eCount == cTarget:
                        lineSylCountListMax.append(eCount)
                        lineSylCountListMin.append(eCount)
                        lineSylCountList = lineSylCountListMax.copy()
                        
                    elif sum(lineSylCountListMin) + eCount == cTarget:
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
        
        if len(wordList) != len(lineSylCountList):
            raise Exception("Word list and syllable counts have different lengths")

        return wordList, lineSylCountList

    def countSyllables(self,wordList):
        # count syllables
        lineSylCountListMin = []
        lineSylCountListMax = []
        for wdx in range(len(wordList)):

            # get word and count
            word = wordList[wdx].lower()
            count = self.snClass.sylDict[word]
            
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

class SonnetWriter(PoemWriter):
    # Ten lines of 14 syllables
    def __init__(self,snClass,hmmClass):
        self.hmmClass = hmmClass
        self.snClass = snClass
        self.poemStructure = []
        for ldx in range(14):
            self.poemStructure.append(10)

class HaikuWriter(PoemWriter):
    # Three lines of 5, 7, 5 syllables
    def __init__(self,snClass,hmmClass):
        self.hmmClass = hmmClass
        self.snClass = snClass
        self.poemStructure = [5,7,5]

class SonnetRhymeWrite(SonnetWriter):
    # Ten lines of 14 syllables
    # Rhyming is ababcdcdefefgg
    def __init__(self,snClass,hmmClass):
        self.hmmClass = hmmClass
        self.snClass = snClass
        self.poemStructure = []
        self.rhymePattern = 'ababcdcdefefgg'
        for ldx in range(14):
            self.poemStructure.append(10)

    def write_poem(self,seed=None):

        # create random number generator
        rng = np.random.default_rng(seed=seed)

        # get rhyming words
        wordRhymeDict = self.snClass.wordRhymeDict
        rhymeKeys = list(wordRhymeDict.keys())

        # randomly choose rhyming words
        thisR1 = rng.random(7) * (len(rhymeKeys)-1)
        rList = []
        for rdx in range(len(thisR1)):
            # get first word
            thisInt = int(thisR1[rdx])
            thisKey = rhymeKeys[thisInt]
            # get second word
            thisEntry = wordRhymeDict.get(thisKey)
            thisR2 = int(rng.random(1) * (len(thisEntry)-1))
            thisMatch = thisEntry[thisR2]
            # store pair of rhyming words
            thisRhyme = [thisKey,thisMatch]
            rList.append(thisRhyme.copy())

        # Sample and convert sentence.
        poemList = []
        failBool = False
        for ldx in range(len(self.poemStructure)):
            
            # get number of syllables
            cTarget = self.poemStructure[ldx]
            
            # get rhyme
            thisRhymeChar = self.rhymePattern[ldx]
            charMatches = [xdx for xdx in range(len(self.rhymePattern)) if self.rhymePattern[xdx] == thisRhymeChar]
            firstBool = charMatches[0] == ldx
            cdx = ord(thisRhymeChar) - ord('a')
            if firstBool:
                lastWord = rList[cdx][0]
            else:
                lastWord = rList[cdx][1]
            
            # get line
            thisLineList, sylList = self.write_line(rng,cTarget,lastWord)

            # ensure that line has the right number of syllables
            lineSylCountListMax, lineSylCountListMin = self.countSyllables(thisLineList)
            cMax = sum(lineSylCountListMax)
            cMin = sum(lineSylCountListMin)
            if cMax == cTarget or cMin == cTarget:
                thisLine = " ".join(thisLineList)
                thisLine = thisLine[0].capitalize() + thisLine[1:]
                poemList.append(thisLine)
            else:
                lineSylCountListMax, lineSylCountListMin = self.countSyllables(thisLineList)
                failBool = True
                raise Exception("Line has wrong number of syllables")
                
        if not failBool:
            for ldx in range(len(poemList)):
                print(poemList[ldx])
                
        return poemList
    
    def write_line(self,rng,cTarget,lastWord):
        
        # get classes
        hmmClass = self.hmmClass
        snClass = self.snClass

        # get last state
        lastObs = snClass.obs_map.get(lastWord)

        modBool = True
        while modBool and lastObs is None:
            if lastWord[-1] == "'":
                lastWord = lastWord[:-1]
                modBool = True
            elif lastWord[0] == "'":
                lastWord = lastWord[1:]
                modBool = True
            else:
                modBool = False
            lastObs = snClass.obs_map.get(lastWord)
        if lastObs is None:
            raise Exception("Observation map is incomplete")
                

        # iterate until appropriate line is written
        lineBool = False
        while lineBool == False:
            # get sample emission
            emission, states = hmmClass.generate_emission_r(cTarget,lastObs,rng)
            thisLine = [snClass.obs_map_r[i] for i in emission]

            # count syllables
            lineSylCountListMin = [0 for _ in range(len(thisLine))]
            lineSylCountListMax = [0 for _ in range(len(thisLine))]
            wordList = ['' for _ in range(len(thisLine))]
            for wdx in range(len(thisLine)-1,0,-1):

                # check if we have enough syllables
                if sum(lineSylCountListMin) > cTarget:
                    lineBool = False
                    break
                elif sum(lineSylCountListMin) == cTarget:
                    lineBool = True
                    lineSylCountList = lineSylCountListMin.copy()
                    break
                elif sum(lineSylCountListMax) == cTarget:
                    lineBool = True
                    lineSylCountList = lineSylCountListMax.copy()
                    break
                word = thisLine[wdx]
                count = snClass.sylDict[word]
                if word == "i":
                    wordList[wdx] = "I"
                else:
                    wordList[wdx] = word

                # check for E
                eBool = False
                countInt = []
                for jdx in range(len(count)):
                    if count[jdx][0] == 'E':
                        eBool = True
                        eCount = int(count[jdx][1])
                    else:
                        countInt.append(int(count[jdx]))
                
                if wdx == len(thisLine) - 1 and eBool:
                    lineSylCountListMax[wdx] = eCount
                    lineSylCountListMin[wdx] = eCount
                else:
                    lineSylCountListMax[wdx] = max(countInt)
                    lineSylCountListMin[wdx] = min(countInt)
        
        # get rid of unneeded word and syls
        wordList = wordList[wdx+1:]
        lineSylCountList = lineSylCountList[wdx+1:]

        if len(wordList) != len(lineSylCountList):
            raise Exception("Word list and syllable counts have different lengths")

        return wordList, lineSylCountList
