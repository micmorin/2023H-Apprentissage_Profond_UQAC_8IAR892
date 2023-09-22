import pickle


class Levels:
    def __init__(self): 
        self.openList()
        
    def getLevelByIndex(self, index):
        return level(self.level_list[index],index)

    def saveList(self):
        with open('levels.pkl', 'wb') as file:
            pickle.dump(self.level_list, file)
        file.close()

    def openList(self):
        with open('levels.pkl', 'rb') as file:
            self.level_list = pickle.load(file)

    def addLevel(self, level_array):
        self.level_list.append(level_array)
        self.saveList()

    def removeLevel(self, level_index):
        self.level_list.pop(level_index)
        self.saveList()


class level:
    def __init__(self, level_array, index):
        self.index = index
        self.name = level_array[0]
        self.hp = level_array[1]
        self.nbr_moves = level_array[2]
        self.disruption_counter = level_array[3]
        self.initial_obstacles = level_array[4]
        self.disruption = level_array[5]
        self.score = 0

    

    """
    Pour l'ajout d'un niveau dans __init__:
        self.addLevel(["caterpie", 1000, 10, 1, [
                                      [0,0,0,0,0,0],
                                      [0,0,0,0,0,0],
                                      [0,0,60,60,0,0],
                                      [0,0,60,60,0,0],
                                      [0,0,0,0,0,0],
                                      [0,0,0,0,0,0] ], [
                                                        [0,0,0,0,0,0],
                                                        [0,0,0,0,0,0],
                                                        [0,0,0,0,0,0],
                                                        [0,0,0,0,0,0],
                                                        [0,0,0,0,0,0],
                                                        [0,0,0,0,0,0] ]])
        """    