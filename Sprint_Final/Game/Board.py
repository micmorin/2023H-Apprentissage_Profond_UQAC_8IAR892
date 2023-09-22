import numpy as np
import pandas as pd
from numpy.random import choice

########
# Board (Obj) [22]
## init(level) [23]
## initialMatrix(level) [29]
### verifyRows() [49]
### verifyColumns() [77]
### addObstructions(disruption_matrix) [93]
## enactMove(move, current_level, user) [109]
### adjustDisruptions(reduce_timer) [131]
### calculateScore(initial_board, current_level, user) [161]
## verifyMove(move) [167]
### verifyOneRow(row, column, value)  [184]
### verifyOneColumn(row, column, value) [196]
## fall() [208]
## printLayout(current_level) [229]
########

class Board(object):
    def __init__(self, level):
        self.level = level
        self.layout = np.zeros([6, 6], dtype = int)
        self.barriers = np.zeros([6, 6], dtype = int)
        self.history = []
        self.initialMatrix(level)  

    def initialMatrix(self, level):
        # Setup Initial Board
        self.layout = choice([1,2,3,4], size=(6, 6))

        # Transform matches to 0s
        self.verifyRows()
        self.verifyColumns()

        # If there are zeros on board, adjust the board and verify matches
        while np.where(self.layout == 0)[0].size != 0:
            self.fall()
            self.verifyRows()
            self.verifyColumns()
        
        # Add level disruptions
        self.addDisruptions(level.initial_obstacles)

        # Print Board
        #self.printLayout(level)
        self.history.append([self.layout,level.hp])

    def verifyRows(self):
        for row in range(6):
            first = self.layout[row, 0]
            second = self.layout[row, 1]

            for column in range(2,6):
                third = self.layout[row, column]            
                
                if first == second == third:        # Verify for a match of 3
                    for i in range(1,5-column):
                        if first == self.layout[row, i]: # Verify for a match of more than 3
                            self.layout[row, i] = 0
                        else: break
                    
                    # A match gets turned to 0s
                    self.layout[row, column]   = 0
                    self.layout[row, column-1] = 0
                    self.layout[row, column-2] = 0

                first = second
                second = third

    def verifyColumns(self):
        for column in range(6):
            first = self.layout[0, column]
            second = self.layout[1, column]

            for row in range(2,6):
                third = self.layout[row, column]            
                
                if first == second == third:        # Verify for a match of 3
                    for i in range(1,5-row):
                        if first == self.layout[i, column]: # Verify for a match of more than 3
                            self.layout[i, column] = 0
                        else: break
                    
                    # A match gets turned to 0s
                    self.layout[row, column]   = 0
                    self.layout[row-1, column] = 0
                    self.layout[row-2, column] = 0

                first = second
                second = third

    def addDisruptions(self, disruption_matrix):
        # Barriers = 10, wood = 20, Metal = 60

        # Take all disruptions but barriers into temp
        temp = np.subtract(disruption_matrix, 10)
        temp[temp<0] = 0

        # Add all disruptions but barriers to board
        self.layout = np.add(self.layout, temp)

        # Put barriers in its own matrix
        for row in range(len(disruption_matrix)):
            for column in range(6):
                if disruption_matrix[row][column] == 10:
                    self.barriers[row][column] = 1

    def enactMove(self, args, current_level, user):
        # Switches both positions values
        start = self.layout[int(args[0]),int(args[1])]
        self.layout[int(args[0]),int(args[1])] = self.layout[int(args[2]),int(args[3])]
        self.layout[int(args[2]),int(args[3])] = start

        # Verify matches and eliminate wood and reduce metal as needed
        copy = self.layout
        self.verifyRows()
        self.verifyColumns()
        self.adjustDisruptions(True)

        # Keep verifying matches until no more
        while np.where(self.layout == 0)[0].size != 0:
            self.calculateScore(copy, current_level, user)
            self.fall()
            self.printLayout(current_level)
            copy = self.layout
            self.verifyRows()
            self.verifyColumns()
            self.adjustDisruptions(False)

    def adjustDisruptions(self, reduce_timer):
        for row in range(6):
            for column in range(6):
                # Remove barriers in a match
                if self.layout[row][column] == 0:
                    self.barriers[row][column] = 0

                    # Remove wood next to a match
                    try: 
                        if int(self.layout[row][column-1]/10) == 1: 
                            self.layout[row][column-1] = 0 
                    except: pass
                    try:
                        if int(self.layout[row][column+1]/10) == 1: 
                            self.layout[row][column+1] = 0
                    except: pass
                    try:
                        if int(self.layout[row-1][column]/10) == 1: 
                            self.layout[row-1][column] = 0
                    except: pass
                    try:
                        if int(self.layout[row+1][column]/10) == 1: 
                            self.layout[row+1][column] = 0
                    except: pass
                
                # Adjust metal as needed
                if reduce_timer:
                    if self.layout[row][column] / 10 > 2: self.layout[row][column] -= 10
                    if self.layout[row][column] / 10 == 2: self.layout[row][column] = 0

    def calculateScore(self, initial_board, current_level, user):
        filter = self.layout == 0 # Create a matrix of boolean where spaces of layout at 0 are true
        removed = initial_board[filter] # Use the filter to extract the values of the matches
        for i in removed: 
            current_level.score += int(user.team[i%10][1]) # Use the extracted values to get the pokemon's power and add to score

    def verifyMove(self, args):
        # Can't move a pokemon in a barrier
        if self.barriers[int(args[0])][int(args[1])] == 1 or self.barriers[int(args[2])][int(args[3])] == 1:
            return False
        
        # Verify a match is present
        if self.verifyOneRow(int(args[0]), int(args[1]), self.layout[int(args[2]), int(args[3])]):
            return True
        elif self.verifyOneRow(int(args[2]), int(args[3]), self.layout[int(args[0]), int(args[1])]):
            return True
        elif self.verifyOneColumn(int(args[0]), int(args[1]), self.layout[int(args[2]), int(args[3])]):
            return True
        elif self.verifyOneColumn(int(args[2]), int(args[3]), self.layout[int(args[0]), int(args[1])]):
            return True
        else:
            return False
           
    def verifyOneRow(self, row, column, value):     
        if column <= 3:
            if value == self.layout[row, column+1] == self.layout[row, column+2]: # Verify a match on Right Right
                return True
        if column >= 2:
            if value == self.layout[row, column-1] == self.layout[row, column-2]: # Verify a match on Left Left
                return True
        if column != 0 and column != 5:
            if value == self.layout[row, column+1] == self.layout[row, column-1]: # Verify a match on Left Right
                return True
        return False
    
    def verifyOneColumn(self, row, column, value):    
        if row <= 3:
            if value == self.layout[row+1, column] == self.layout[row+2, column]: # Verify a match on Down Down
                return True
        if row >= 2:
           if value == self.layout[row-1, column] == self.layout[row-2, column]: # Verify a match on Up Up
                return True
        if row != 0 and row != 5:
            if value == self.layout[row+1, column] == self.layout[row-1, column]: # Verify a match on Up Down
                return True
        return False
   
    def fall(self):
        for row in range(5,0,-1):
            for column in range(6):
                # If there was a match
                if self.layout[row,column] == 0:
                    # Iterate from the row above to the top
                    for i in range(row-1,-1,-1):
                        # If the top row is not 0, switch the values
                        if self.layout[i,column] != 0: 
                            self.layout[row,column] = self.layout[i,column]
                            self.layout[i,column] = 0
                            break
                        # If we went trough every row above and they were all zero, generate a new value for the match
                        if i == 0: 
                            self.layout[row,column] = choice([1,2,3,4])

        # For the top row, matches get a newly generated value                
        for column in range(6):
                if self.layout[0,column] == 0:
                    self.layout[0,column] = choice([1,2,3,4])

    def printLayout(self, current_level):
        
        # Print the game and barrier board
        print("\n    Game ", "       Barriers")
        print(self.layout[0,:],self.barriers[0,:])
        print(self.layout[1,:],self.barriers[1,:])
        print(self.layout[2,:],self.barriers[2,:])
        print(self.layout[3,:],self.barriers[3,:])
        print(self.layout[4,:],self.barriers[4,:])
        print(self.layout[5,:],self.barriers[5,:])

        # Print the current HP
        current_level.hp = current_level.hp - current_level.score
        if current_level.hp < 0: current_level.hp = 0
        print("HP:"+str(current_level.hp))

        self.history.append([self.layout,current_level.hp])
