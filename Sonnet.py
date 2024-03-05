import re

class Sonnet:
    def __init__(self,filename,sequenceType,sylDict):
        self.filename = filename
        self.sequenceType = sequenceType
        self.sonnetLengthList = [12,13,14,15]
        self.sylDict = sylDict

    def read(self):

        # open file
        f = open(self.filename,"r")

        # count and store sonnets
        sonnetList = []
        thisSonnet = []
        sCount = 0

        # iterate through sonnets
        for x in f:

            # determine whether number, line, or blank space
            thisStr = x[0:-1]
            thisWC = len(thisStr.split())

            # determine whether new sonnet
            newSonnet = False                
            if thisWC == 0:
                if len(thisSonnet) in self.sonnetLengthList:
                    newSonnet = True
                else:
                    thisSonnet = []

            if newSonnet: # number
                thisLen = len(thisSonnet)
                if thisLen in self.sonnetLengthList:
                    sonnetList.append(thisSonnet.copy())
                else:
                    print("Length not supported")
                sCount += 1
                thisSonnet = []            
            elif thisWC > 1: # line
                thisStr = x[0:-1]
                thisList = thisStr.split()
                thisSonnet.append(thisList.copy())

        # handle final sonnet
        if len(thisSonnet) in self.sonnetLengthList:
            sonnetList.append(thisSonnet.copy())
        self.sonnetList = sonnetList
        pass
    
    def buildSequenceStr(self):
        if self.sequenceType == "Stanza":
            self.sequenceStanzaStr()
        else:
            print("Sequence type not specified")

    def sequenceStanzaStr(self):
        """
        Group sequences by stanza
        13 lines: 3 + 4 + 4 + 2
        14 lines: 4 + 4 + 4 + 2
        15 lines: 5 + 4 + 4 + 2
        12 lines: 2 + 2 + 2 + 2 + 2 + 2
        """
        sequenceList = []

        # iterate through sonnets
        for sdx in range(len(self.sonnetList)):
            
            # get structure of sonnets
            thisLen = len(self.sonnetList[sdx])           
            if thisLen == 12:
                iList = [0,2,4,6,8,10,12]
            elif thisLen == 13:
                iList = [0,3,7,11,13]
            elif thisLen == 14:
                iList = [0,4,8,12,14]     
            elif thisLen == 15:
                iList = [0,5,9,13,15]
            else:
                print("Sonnet length not handled properly: ", thisLen)
            
            # iterate through stanzas
            for idx in range(len(iList)-1):
                thisStanza = []
                iStart = iList[idx]
                iStop = iList[idx+1]

                # iterate through lines
                for jdx in range(iStart,iStop):
                    thisStanza.append(self.sonnetList[sdx][jdx].copy())
                
                # store stanza in sequence list
                sequenceList.append(thisStanza)

        self.sequenceListStr = sequenceList
        pass

    def parse_observations(self):

        obs_counter = 0
        obs = []
        obs_map = {}

        # iterate through stanzas
        for sdx in range(len(self.sequenceListStr)):
            obs_elem = []
            thisStanza = self.sequenceListStr[sdx]

            # iterate through lines in stanza
            for ldx in range(len(thisStanza)):
                lineSylCountListMin = []
                lineSylCountListMax = []
                wordList = []

                # iterate through words in stanza
                for wdx in range(len(thisStanza[ldx])):
                    
                    # last word in line?
                    endBool = wdx == (len(thisStanza[ldx]) - 1)

                    # get word
                    word = thisStanza[ldx][wdx].lower()
                    if endBool and word[-1] == "'":
                        word = word[0:-1]
                    if word[-1] in [',','.',':',';','?','!','.']:
                        word = word[0:-1]

                    # get syllable count for word
                    try:
                        thisDict = self.sylDict[word]
                    except:
                        modBool = True
                        dictReadFailBool = True
                        while modBool and dictReadFailBool:
                            if word[0] == "'":
                                modBool = True
                                word = word[1:]
                            elif word[0] == "(":
                                modBool = True
                                word = word[1:]
                            elif word[-1] == ")":
                                modBool = True
                                word = word[:-1]
                            elif word[-1] == "'":
                                modBool = True
                                word = word[:-1]
                            else:
                                modBool = False
                            if modBool:
                                try:
                                    thisDict = self.sylDict[word]
                                    dictReadFailBool = False
                                except:
                                    dictReadFailBool = True
                                if not modBool:
                                    print("Failed to read word from dictionary")
                            else:
                                print("Failed to read word from dictionary")
                            
                    if len(thisDict) == 1:
                        sylCountList = [int(thisDict[0])]
                    else:
                        sylCountList = []

                        # get possible list of syllable counts
                        for ddx in range(len(thisDict)):
                            if thisDict[ddx][0] == 'E':
                                if endBool:
                                    sylCount = int(thisDict[ddx][1])
                                    sylCountList = [sylCount]
                                    break
                            else:
                                sylCountList.append(int(thisDict[ddx][0]))

                    # save word
                    wordList.append(word)

                    # get syllable count (for line)
                    if len(sylCountList) == 1:
                        sylCount = sylCountList[0]
                        lineSylCountListMin.append(sylCount)
                        lineSylCountListMax.append(sylCount)                   
                    else:
                        lineSylCountListMax.append(max(sylCountList))
                        lineSylCountListMin.append(min(sylCountList))
                    
                    # check syllable count
                    if sum(lineSylCountListMax) == 10:
                        lineSylCountList = lineSylCountListMax.copy()
                    elif sum(lineSylCountListMin) == 10:
                        lineSylCountList = lineSylCountListMin.copy()
                    else:
                        if sum(lineSylCountListMin) > 10:
                            lineSylCountList = lineSylCountListMin.copy()
                        elif sum(lineSylCountListMax) < 10:
                            lineSylCountList = lineSylCountListMax.copy()
                        else:
                            # count being and whether
                            beingIndexList = []
                            whetherIndexList = []
                            crownedIndexList = []
                            flowerIndexList = []
                            for wdx in range(len(thisStanza[ldx])):
                                word = re.sub(r'[^\w]', '', thisStanza[ldx][wdx]).lower()
                                if word == 'being':
                                    beingIndexList.append(wdx)
                                elif word == 'whether':
                                    whetherIndexList.append(wdx)
                                elif word == 'crowned':
                                    crownedIndexList.append(wdx)
                                elif word == 'flowers':
                                    flowerIndexList.append(wdx)

                            if len(beingIndexList) in [1,2] and sum(lineSylCountListMax) == 11: # handle being
                                lineSylCountList = lineSylCountListMax.copy()
                                lineSylCountList[beingIndexList[0]] = 1
                            elif len(whetherIndexList) in [1,2] and sum(lineSylCountListMax) == 11: # handle whether
                                lineSylCountList = lineSylCountListMax.copy()
                                lineSylCountList[whetherIndexList[0]] = 1
                            elif len(crownedIndexList) == 1 and len(whetherIndexList) == 1 and len(beingIndexList) == 1 and sum(lineSylCountListMax) == 12:
                                lineSylCountList = lineSylCountListMax.copy()
                                lineSylCountList[crownedIndexList[0]] = 1
                                lineSylCountList[beingIndexList[0]] = 1
                            elif len(flowerIndexList) in [1,2] and sum(lineSylCountListMax) == 11: # handle flower
                                lineSylCountList = lineSylCountListMax.copy()
                                lineSylCountList[flowerIndexList[0]] = 1
                            else:
                                print("Problem with syllable count")
                    
                # iterate through words in stanza
                for wdx in range(len(thisStanza[ldx])):
                    
                    # get word and its syllable count
                    word = wordList[wdx]
                    thisCount = lineSylCountList[wdx]
                    thisObs = word# + str(thisCount)
                    
                    if thisObs not in obs_map:
                        # Add unique words to the observations map.
                        obs_map[thisObs] = obs_counter
                        obs_counter += 1

                    # Add the encoded word
                    obs_elem.append(obs_map[thisObs])
                
                # Add the encoded sequence
                obs.append(obs_elem)

        # store values
        self.obs = obs
        self.obs_map = obs_map
        self.obs_counter = obs_counter
        pass
    
    def obs_map_reverser(self):
        obs_map_r = {}

        for key in self.obs_map:
            obs_map_r[self.obs_map[key]] = key

        self.obs_map_r = obs_map_r
        pass