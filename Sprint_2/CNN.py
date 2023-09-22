#---import the modules---
from os.path import isfile, normpath
from shutil import rmtree
from sys import stdout
import numpy as np
from keras.utils import load_img, img_to_array, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras_tuner import RandomSearch, HyperModel
import pandas as pd 
import json

class CNN(HyperModel):
    path = './'

    def generateDataset(self):
        datagenerator = ImageDataGenerator(width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            horizontal_flip=True,
                                            brightness_range=[0.5,1.5],
                                            zoom_range=[1.5,0.5]
                                            )
        
        iterator = datagenerator.flow_from_directory(self.path+'images',
                                                    target_size=(32,32), 
                                                    color_mode='rgb', 
                                                    batch_size=258, 
                                                    class_mode='sparse', 
                                                    shuffle=True)

        dataset = []

        for i in range(1):
            batch = iterator.next()

            x = np.reshape(batch[0],(258,3072))
            y = np.reshape(batch[1],(258,1))

            temp_array = np.append(x, y, axis=1)
            if i == 0: dataset = temp_array
            else: dataset = np.append(dataset, temp_array, axis=0)

            stdout.write("Generating Dataset "+str(round(i/300*100,2))+"%\r")
            stdout.flush()

        pd.DataFrame(dataset).to_csv(self.path+"CNN_Model/dataset.csv", index=False)

    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)

        # Generate dataset if necessary
        if not isfile(self.path+'CNN_Model/dataset.csv'):
            self.path = './Pokemon/Sprint_2/'
            if not isfile(self.path+'CNN_Model/dataset.csv'):
                self.generateDataset()

        # Open dataset
        dataset = pd.read_csv(self.path+'CNN_Model/dataset.csv').to_numpy()
        self.x_train = np.reshape(dataset[:,:-1],(dataset.shape[0],32,32,3))
        self.y_train = np.reshape(dataset[:,-1:],(dataset.shape[0]))

        # Open validation
        self.x_test = [img_to_array(load_img(self.path+'static/Barrier.png'))]
        self.x_test.append(img_to_array(load_img(self.path+'static/Bulbasaur.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Caterpie.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Charmander.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Eevee.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Metal.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Pidgey.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Pikachu.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Squirtle.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Weedle.png')))
        self.x_test.append(img_to_array(load_img(self.path+'static/Wood.png')))
        self.x_test = np.array(self.x_test)
        self.y_test = np.reshape([0,1,2,3,4,5,6,7,8,9,10],(11,1))

        # Open Model
        if isfile(self.path+'CNN_Model/model.json') and isfile(self.path+'CNN_Model/model.json'):
            self.model = self.loadModel()
        else:
            print("Generating CNN model #1")
            self.model = self.build_fast()
            count = 1
            # If model is less than 90% accurate, loop until one does
            while(self.model.evaluate(self.x_test, self.y_test, return_dict=True)["accuracy"] < .9):
                count = count+ 1
                print("Generating CNN model #"+str(count))
                try: rmtree(self.path+'CNN_Model/Tuner_trials') # Enables buil instead of build_fast
                except FileNotFoundError: pass
                self.model = self.build_fast()
            
            # Save Model    
            f = open(self.path+'CNN_Model/model.json', 'w')
            f.write(self.model.to_json())
            f.close()
            self.model.save_weights(self.path+'CNN_Model/model.h5')

    def build(self, kerasTuner, *args, **kwargs):
        # Build Hypermodel for tuner
        model = models.Sequential()
        model.add(layers.Conv2D(
                filters=kerasTuner.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                kernel_size=3,
                activation='relu',
                input_shape=(32,32,3)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(
                filters=kerasTuner.Int('conv_2_filter', min_value=32, max_value=128, step=16),
                kernel_size=3,
                activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(
                filters=kerasTuner.Int('conv_3_filter', min_value=32, max_value=128, step=16),
                kernel_size=3,
                activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(
                units=kerasTuner.Int('dense_1_filter', min_value=32, max_value=128, step=16),
                activation='relu'
            ))
        model.add(layers.Dense(11, activation='softmax'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        return model
    
    def build_fast(self):
        # Build model based on the one time I got a model at 100% accuracy
        model = models.Sequential()
        model.add(layers.Conv2D(
                filters=64,
                kernel_size=3,
                activation='relu',
                input_shape=(32,32,3)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(
                filters=80,
                kernel_size=3,
                activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(
                filters=80,
                kernel_size=3,
                activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(
                units=96,
                activation='relu'
            ))
        model.add(layers.Dense(11, activation='softmax'))
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, epochs=30, validation_data=(self.x_test, self.y_test), verbose=0)
        return model

    def fit(self, kerasTuner, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=kerasTuner.Int("batch_size", min_value=1, max_value=258, step=1),
            **kwargs,
        )
    
    def toText(self, max_trial, tuner):
        # Creates a list of all model's accuracy in ascending order
        result = []
        max_zeros = len(list(str(max_trial)))

        # Opening JSON file
        for i in range(max_trial):
            s = (max_zeros-len(list(str(i))))*'0'+str(i)
            if isfile(self.path+'CNN_Model/Tuner_trials/trial_'+s+'/trial.json'):
                f = open(self.path+'CNN_Model/Tuner_trials/trial_'+s+'/trial.json')
                data = json.load(f)
                result.append({'s':s,
                    'acc':data['score'],
                    'struct':data['hyperparameters']['values']})
                f.close()

        result = sorted(result, key=lambda d: d['acc'])
        f = open(self.path+'CNN_Model/results.txt', 'w')
        f.write(json.dumps(result,indent=1))
        f.close()

    def createModel(self):
        max_trial = 10
        tuner = RandomSearch(hypermodel=self.build_fast,
                        objective='val_accuracy',
                        max_trials = max_trial,
                        directory=normpath(self.path),
                        project_name='CNN_Model/Tuner_trials')

        tuner.search(self.x_train, self.y_train, epochs=30, validation_data=(self.x_test, self.y_test), verbose=2)
        self.toText(max_trial, tuner)
        self.model = tuner.get_best_models(num_models=1)[0]

    def loadModel(self):
        f = open(self.path+'CNN_Model/model.json', 'r')
        loaded_model = models.model_from_json(f.read())
        f.close()

        loaded_model.load_weights(self.path+"CNN_Model/model.h5")
    
        loaded_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return loaded_model
    
    def getMove(self, board_object):
        self.visualMatrixToMatrixForAgent(board_object)
        self.recordVisualizedMatrix(board_object)
        #return agent.getmove(self.matrix)

    def visualMatrixToMatrixForAgent(self, board_object, user):
        # Initialize variables
        self.matrix = np.zeros([6, 6], dtype = int)
        self.assertTeam(user)
        
        # Iterate over layout matrix
        for row_number in range(len(board_object.layout)):
            for value_index in range(len(board_object.layout[row_number])):
                image_path = ""
                value = board_object.layout[row_number][value_index]

                # Find the image to decode
                if board_object.barriers[row_number][value_index] == 1:
                    image_path = 'static/'+str(user.team[value-1][0])+'_b.png'
                elif value < 5:
                    image_path = 'static/'+str(user.team[value-1][0])+'.png'
                elif value < 20:
                    image_path = 'static/Wood.png'
                else:
                    image_path = 'static/Metal.png' 

                # Get image in array form
                img = img_to_array(load_img(self.path+image_path))

                # Decode image
                prediction = self.model.predict(img.reshape((1,32,32,3)), verbose=0).argmax()

                # Adjust the 0-10 categories to reflect current team
                if prediction == 0 or prediction == 5 or prediction == 10:
                    if prediction == 5: prediction = 20
                    elif prediction == 10: prediction = 40
                else:
                    for p in range(len(self.team)):
                        if prediction == self.team[p]: 
                            prediction = p+1
                            break


                self.matrix[row_number][value_index] = prediction

    def assertTeam(self, user):
        # Helps correcting the 0-10 categories toward current team
        self.team = []
        for p in user.getTeam():
            if p == "Bulbasaur" :  self.team.append(1)
            elif p == "Caterpie" :  self.team.append(2)
            elif p == "Charmander" :  self.team.append(3)
            elif p == "Eevee" :  self.team.append(4)
            elif p == "Pidgey" :  self.team.append(6)
            elif p == "Pikachu" :  self.team.append(7)
            elif p == "Squirtle" :  self.team.append(8)
            elif p == "Weedle" :  self.team.append(9)

    def recordVisualizedMatrix(self, board_object):
        board_object.history[len(board_object.history)-1].append(self.matrix)

