import re

class Sonnet:
    def __init__(self,filename,sequenceType):
        self.filename = filename
        self.sequenceType = sequenceType
        self.sonnetLengthList = [12,13,14,15]

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
                thisStr = x[0:-2]
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
                    thisStanza.extend(self.sonnetList[sdx][jdx])
                
                # store stanza in sequence list
                sequenceList.append(thisStanza)

        self.sequenceListStr = sequenceList
        pass

    def parse_observations(self):

        obs_counter = 0
        obs = []
        obs_map = {}
        for sdx in range(len(self.sequenceListStr)):
            obs_elem = []

            for jdx in range(len(self.sequenceListStr[sdx])):
                word = self.sequenceListStr[sdx][jdx]
                word = re.sub(r'[^\w]', '', word).lower()
                if word not in obs_map:
                    # Add unique words to the observations map.
                    obs_map[word] = obs_counter
                    obs_counter += 1

                # Add the encoded word
                obs_elem.append(obs_map[word])
            
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