class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop() #self.category = standing or jumbing or ...
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1,y1,x2,y2,fram_ID,lost,grouping,generated= words
        self.box = x1,y1,x2,y2
        self.frame_ID = fram_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated
