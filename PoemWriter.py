import numpy as np

class PoemWriter:
    def __init__(self,snClass,hmmClass):
        self.hmmClass = hmmClass
        self.snClass = snClass
        self.poemStructure = []

    def write_poem(self,seed=None):

        # create random number generator
        rng = np.random.default_rng(seed=seed)

        # include period?
        perBool1 = type(self).__name__ == 'SonnetWriter'
        perBool2 = type(self).__name__ == 'SonnetStressWriter'
        perBool = perBool1 or perBool2

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
                if ldx == len(self.poemStructure)-1 and perBool:
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

        # store word and syllable
        lineSylCountList = []
        wordList = []
        e0 = None
        s0 = None
        emission = []
        states = []

        # iterate until appropriate line is written
        lineBool = False
        while lineBool == False:
            
            # check if we have enough syllables
            if sum(lineSylCountList) > cTarget or np.isnan(sum(lineSylCountList)):
                lineSylCountList = []
                wordList = []
                e0 = None
                s0 = None
                emission = []
                states = []
            elif sum(lineSylCountList) == cTarget:
                lineBool = True
                break


            # generate new word
            e1, s1 = hmmClass.generate_emission(1,rng,e0,s0)
            thisWord = snClass.obs_map_r[e1[-1]]

            # append new word
            emission.append(e1[-1])
            states.append(s1[-1])
            e0 = e1[-1]
            s0 = s1[-1]
            wordList.append(thisWord)
            count = snClass.sylDict[thisWord]
            if len(count) == 1:
                lineSylCountList.append(int(count[0]))
            else:
                # check for E
                eBool = False
                countInt = []
                for jdx in range(len(count)):
                    if count[jdx][0] == 'E':
                        eBool = True
                        eCount = int(count[jdx][1])
                    else:
                        countInt.append(int(count[jdx]))
                
                # get thisCount
                if len(countInt) == 1:
                    thisCount = countInt[0]
                else:
                    thisR = rng.random(1)
                    pCumSum1 = np.cumsum(np.ones(len(countInt)) / len(countInt))
                    pCumSum = np.zeros(len(countInt)+1)
                    pCumSum[1:] = pCumSum1
                    for jdx in range(len(countInt)):
                        if thisR > pCumSum[jdx] and thisR <= pCumSum[jdx+1]:
                            thisCount = countInt[jdx]
                
                # handle end
                if eBool:
                    if sum(lineSylCountList) + eCount == cTarget:
                        lineSylCountList.append(eCount)
                    elif sum(lineSylCountList) + thisCount >= cTarget:
                        lineSylCountList.append(np.nan)
                    else:
                        lineSylCountList.append(thisCount)
                else:
                    lineSylCountList.append(thisCount)                      
                        
                                    
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

class SonnetRhymeWriter(SonnetWriter):
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

        # get number of rhyming words
        rhymeCountList = []
        rhymeCharList = []
        for rdx in range(len(self.rhymePattern)):
            if self.rhymePattern[rdx] in rhymeCharList:
                matches = [xdx for xdx in range(len(rhymeCharList)) if rhymeCharList[xdx] == self.rhymePattern[rdx]]
                if len(matches) != 1:
                    raise Exception("Rhyming not handled properly")
                rhymeCountList[matches[0]] += 1
            else:
                rhymeCharList.append(self.rhymePattern[rdx])
                rhymeCountList.append(1)

        # randomly choose rhyming words
        rList = []
        for rdx in range(len(rhymeCountList)):
            # get list of rhyming words
            thisEntry = []
            while len(thisEntry) < rhymeCountList[rdx] - 1:
                # get first word
                thisR1 = rng.random(1) * (len(rhymeKeys)-1)
                thisInt = int(thisR1)
                thisKey = rhymeKeys[thisInt]
                thisEntry = wordRhymeDict.get(thisKey).copy()

            # get subsequent words
            thisRhyme = [thisKey]
            for mdx in range(rhymeCountList[rdx]-1):
                thisR2 = rng.random(1) * (len(thisEntry)-1)
                thisInt = int(thisR2)
                thisMatch = thisEntry[thisInt]
                thisRhyme.append(thisMatch)
                thisEntry.remove(thisMatch)
            rList.append(thisRhyme.copy())

        # Sample and convert sentence.
        poemList = []
        failBool = False
        for ldx in range(len(self.poemStructure)):
            
            # get number of syllables
            cTarget = self.poemStructure[ldx]
            
            # get rhyme
            thisRhymeChar = self.rhymePattern[ldx]
            cdx = ord(thisRhymeChar) - ord('a')
            lastWord = rList[cdx][0]
            rList[cdx].remove(lastWord)

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

        # add a period
        poemList[-1] += "."

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
            emission, states = hmmClass.generate_emission_r(cTarget,lastObs,None,rng)
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

class SonnetStressWriter(SonnetWriter):
    def write_line(self,rng,cTarget):
        
        # get classes
        hmmClass = self.hmmClass
        snClass = self.snClass

        # store word and syllable
        lineSylCountList = []
        wordList = []
        e0 = None
        s0 = None
        emission = []
        states = []
        stressList = []

        # iterate until appropriate line is written
        lineBool = False
        while lineBool == False:
            
            # check if we have enough syllables
            if sum(lineSylCountList) > cTarget or np.isnan(sum(lineSylCountList)):
                lineSylCountList = []
                wordList = []
                e0 = None
                s0 = None
                emission = []
                states = []
                stressList = []
            elif sum(lineSylCountList) == cTarget:
                lineBool = True
                break

            # check stress
            if sum(lineSylCountList) % 2 == 0:
                stressBool = False
            else:
                stressBool = True

            # get sample emission
            appendBool = False
            while not appendBool:

                # generate new word
                e1, s1 = hmmClass.generate_emission(1,rng,e0,s0)
                thisWord = snClass.obs_map_r[e1[-1]]

                # should we append new word
                if stressBool and int(thisWord[-1]) == 1:
                    appendBool = True
                elif not stressBool and int(thisWord[-1]) == 0:
                    appendBool = True
                else:
                    appendBool = False

                # append new word
                if appendBool:
                    emission.append(e1[-1])
                    states.append(s1[-1])
                    e0 = e1[-1]
                    s0 = s1[-1]
                    wordList.append(thisWord[:-1])
                    stressList.append(int(thisWord[-1]))
                    count = snClass.sylDict[thisWord[:-1]]
                    if len(count) == 1:
                        lineSylCountList.append(int(count[0]))
                    else:
                        # check for E
                        eBool = False
                        countInt = []
                        for jdx in range(len(count)):
                            if count[jdx][0] == 'E':
                                eBool = True
                                eCount = int(count[jdx][1])
                            else:
                                countInt.append(int(count[jdx]))
                        
                        # get thisCount
                        if len(countInt) == 1:
                            thisCount = countInt[0]
                        else:
                            thisR = rng.random(1)
                            pCumSum1 = np.cumsum(np.ones(len(countInt)) / len(countInt))
                            pCumSum = np.zeros(len(countInt)+1)
                            pCumSum[1:] = pCumSum1
                            for jdx in range(len(countInt)):
                                if thisR > pCumSum[jdx] and thisR <= pCumSum[jdx+1]:
                                    thisCount = countInt[jdx]
                        
                        # handle end
                        if eBool:
                            if sum(lineSylCountList) + eCount == cTarget:
                                lineSylCountList.append(eCount)
                            elif sum(lineSylCountList) + thisCount >= cTarget:
                                lineSylCountList.append(np.nan)
                            else:
                                lineSylCountList.append(thisCount)
                        else:
                            lineSylCountList.append(thisCount)                      

        # check lengths
        if len(wordList) != len(lineSylCountList):
            raise Exception("Word list and syllable counts have different lengths")
        if len(wordList) != len(stressList):
            raise Exception("Word list and stress lists have different lengths")

        # check stress
        stressFailBool = False
        if stressList[0] == 1:
            stressFailBool = True
        for wdx in range(1,len(wordList)):
            if lineSylCountList[wdx-1] % 2 == 0:
                if stressList[wdx] != stressList[wdx-1]:
                    stressFailBool = True
            else:
                if stressList[wdx] == stressList[wdx-1]:
                    stressFailBool = True
        
        if stressFailBool:
            raise Exception("Stresses not correct")
        
        return wordList, lineSylCountList

class SonnetRhymeStressWriter(SonnetRhymeWriter):
    def write_line(self,rng,cTarget,lastWord):
        
        # get classes
        hmmClass = self.hmmClass
        snClass = self.snClass

        # get syllable count
        count = snClass.sylDict.get(lastWord)
        eBool = False
        countList = []
        for thisCount in count:
            if thisCount[0] == 'E':
                eCount = int(thisCount[1])
                eBool = True
            else:
                countList.append(int(thisCount[0]))

        stressBool1 = False
        stressBool0 = False
        if eBool:
            if eCount % 2 == 0:
                stressBool0 = True
                lastCount0 = eCount
            else:
                stressBool1 = True
                lastCount1 = eCount
        else:
            for jdx in countList:
                if jdx % 2 == 0:
                    stressBool0 = True
                    lastCount0 = int(jdx)
                else:
                    stressBool1 = True
                    lastCount1 = int(jdx)

        # get last obs
        lastObs1 = snClass.obs_map.get(lastWord+"1")
        lastObs0 = snClass.obs_map.get(lastWord+"0")
        modBool = True
        failBool0 = lastObs0 is None
        failBool1 = lastObs1 is None
        while modBool and failBool0 and failBool1:
            if lastWord[-1] == "'":
                lastWord = lastWord[:-1]
                modBool = True
            elif lastWord[0] == "'":
                lastWord = lastWord[1:]
                modBool = True
            else:
                modBool = False
            lastObs1 = snClass.obs_map.get(lastWord+"1")
            lastObs0 = snClass.obs_map.get(lastWord+"0")
            failBool0 = lastObs0 is None
            failBool1 = lastObs1 is None
        
        if not lastObs0 is None and stressBool0:
            stressBool0 = True
        elif not lastObs1 is None and stressBool1:
            stressBool1 = True
        
        if stressBool1 and stressBool0:
            thisR = rng.random(1)
            if thisR > 0.0 and thisR <= 0.5:
                lastObs = lastObs0
                lastWord = lastWord + "0"
                lastCount = lastCount0
            else:
                lastObs = lastObs1
                lastWord = lastWord + "1"
                lastCount = lastCount1
        elif stressBool0:
            lastObs = lastObs0
            lastWord = lastWord + "0"
            lastCount = lastCount0
        elif stressBool1:
            lastObs = lastObs1
            lastWord = lastWord + "1"
            lastCount = lastCount1
        else:
            raise Exception("Observation map is incomplete")
            
        
        # store word and syllable
        wordList = ["" for _ in range(cTarget)]
        stressList = [np.nan for _ in range(cTarget)]
        emission = [np.nan for _ in range(cTarget)]
        lineSylCountList = [np.nan for _ in range(cTarget)]
        states = [np.nan for _ in range(cTarget)]
        wordList[-1] = lastWord[:-1]
        stressList[-1] = int(lastWord[-1])
        lineSylCountList[-1] = lastCount
        e0 = lastObs
        s0 = None
        wdx = len(wordList) - 1
        
        # iterate until appropriate line is written
        lineBool = False        
        while lineBool == False:
            
            # do we have enough syllables?
            thisSum = sum(lineSylCountList[wdx:])
            if thisSum == cTarget:
                lineBool = True
                break
            elif thisSum > cTarget:
                wordList = ["" for _ in range(cTarget)]
                stressList = [np.nan for _ in range(cTarget)]
                emission = [np.nan for _ in range(cTarget)]
                lineSylCountList = [np.nan for _ in range(cTarget)]
                states = [np.nan for _ in range(cTarget)]
                wordList[-1] = lastWord[:-1]
                stressList[-1] = int(lastWord[-1])
                lineSylCountList[-1] = lastCount
                e0 = lastObs
                s0 = None
                wdx = len(wordList) - 1
                
            # update counter
            wdx -= 1

            # should next syllable be stressed?
            thisSum = sum(lineSylCountList[wdx+1:])
            if np.isnan(thisSum):
                raise Exception("Error in syllable sum")
            if thisSum % 2 == 1:
                endStressBool = True
            else:
                endStressBool = False

            # get sample emission
            appendBool = False
            while not appendBool:

                # get sample word
                e1, s1 = hmmClass.generate_emission_r(1,e0,s0,rng)
                thisWord = snClass.obs_map_r[e1[0]]

                # get syllable count
                count = snClass.sylDict.get(thisWord[:-1])
                countList = []
                for thisCount in count:
                    if not thisCount[0] == 'E':
                        countList.append(int(thisCount[0]))
                    
                # get syllable count
                thisR = rng.random(1)
                pCumSum1 = np.cumsum(np.ones(len(countList)) / len(countList))
                pCumSum = np.zeros(len(countList)+1)
                pCumSum[1:] = pCumSum1
                for jdx in range(len(countList)):
                    if thisR > pCumSum[jdx] and thisR <= pCumSum[jdx+1]:
                        thisCount = countList[jdx]

                # should start of word be stressed?
                if thisCount % 2 == 0:
                    stressBool = endStressBool
                else:
                    stressBool = not endStressBool

                # should word be appended?
                if stressBool and int(thisWord[-1]) == 1:
                    appendBool = True
                elif not stressBool and int(thisWord[-1]) == 0:
                    appendBool = True
                else:
                    appendBool = False
                
                if appendBool:

                    lineSylCountList[wdx] = thisCount
                    wordList[wdx] = thisWord[:-1]
                    stressList[wdx] = int(stressBool)
                    emission[wdx+1] = e1[-1]
                    states[wdx+1] = s1[-1]
                    emission[wdx] = e1[0]
                    states[wdx] = s1[0]
                    e0 = e1[0]
                    s0 = s1[0]
                            
        # get rid of unneeded word and syls
        wordList = wordList[wdx:]
        lineSylCountList = lineSylCountList[wdx:]
        stressList = stressList[wdx:]

        # check lengths
        if len(wordList) != len(lineSylCountList):
            raise Exception("Word list and syllable counts have different lengths")
        if len(wordList) != len(stressList):
            raise Exception("Word list and stress lists have different lengths")

        # check stress
        stressFailBool = False
        if stressList[0] == 1:
            stressFailBool = True
        for wdx in range(1,len(wordList)):
            if lineSylCountList[wdx-1] % 2 == 0:
                if stressList[wdx] != stressList[wdx-1]:
                    stressFailBool = True
            else:
                if stressList[wdx] == stressList[wdx-1]:
                    stressFailBool = True
        
        if stressFailBool:
            raise Exception("Stresses not correct")
        return wordList, lineSylCountList
    
class LimerickWriter(SonnetRhymeWriter):
    # Five lines [9,9,6,6,9]
    # Rhyming is aabba
    def __init__(self,snClass,hmmClass):
        self.hmmClass = hmmClass
        self.snClass = snClass
        self.poemStructure = [9,9,6,6,9]
        self.rhymePattern = 'aabba'

class PetrarchanSonnetWriter(SonnetRhymeWriter):
    # 14 lines of 11 syllables
    # Rhyming is abbaabbacdecde
    def __init__(self,snClass,hmmClass):
        self.hmmClass = hmmClass
        self.snClass = snClass
        self.poemStructure = []
        self.rhymePattern = 'abbaabbacdecde'
        for ldx in range(14):
            self.poemStructure.append(11)