from utils import *
import os
import numpy as np 
import pandas as pd
from sqlalchemy import create_engine
import keras as K

class PX4Generator(K.utils.Sequence):

    def __init__(self,train:bool=False,**kwargs):
        # Load the CSV data here maybe using pandas dataframe
        # merge and sort them all together
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.batch_size = 64
        self.time_steps = 15        
        self.validation_split = 0.3
        self.train = train
        os.path.basename
        self.path = os.path.join(self.dir ,'log_files/log_07_23/csv_files/')
        self.db_path = os.path.join(self.dir,"data")
        self.db_name = "data"+str(os.path.basename(os.path.dirname(os.path.dirname(self.path))))[-6:]+".sqlite"
        self.tbl_name = "data"
        self.data = list()
        #self.idx = {0:0}
        self.__dict__.update(kwargs)
        

        
        if not os.path.exists(self.db_path):
            csv2sqlite(self.path,self.db_path,self.db_name,self.tbl_name)
        else:
            if self.db_name not in os.listdir(self.db_path):
                csv2sqlite(self.path,self.db_path,self.db_name,self.tbl_name)
            

        #print("loading the SQLite file...")
        #df = self.load_sqlite(self.db_path,self.db_name,self.tbl_name)
        #print("Completed!")

        
        print("loading the dataset...")
        df = load_n_debug(self.path)   
        print("Completed!")
        print("Normalizing the dataset...")
        df = (df - np.min(df, axis = 0)) / (np.max(df,axis = 0) - np.min(df, axis = 0 ))
        df = df.fillna(value=0)
        print("completed!")
        
        end =  len(df)
        idx = int(end * self.validation_split)
        if train:
            print("loading training data...")
            start = 0
            end -= idx
        else:
            print("loading validation data...")
            start = end - idx  
        print("Completed!")

        self.data = df.values[start:end]
        self.num_cols = self.data.shape[1]

        

    def load_sqlite(self, db_path, db_name = "data.sqlite", tbl_name = "data"):
        """Reads the sqlite file in the given "db_path" and converts it to pandas.DataFrame.

        Argument:
            db_path: Absolute path of the sqlite database
            db_name: Name of the database
            tbl_name: Name of the database table
        
        Returns:
            A database as a pandas.DataFrame object

        """

        engine = create_engine("sqlite:///" + os.path.join(db_path,db_name))  # Creating the engine

        query = "SELECT * FROM "+ tbl_name  # String containing the SQL query to select all rows
        
        dataframe = pd.read_sql_query(query, engine)  # Finally, importing the data into DataFrame df
        
        return dataframe
    

    def group_data(self, start_idx, end_idx):
        """trims the values of the data between given thresholds.

        Arguments:
            start_idx: Minimum threshold 
            end_idx: Maximum threshold

        Returns:
            A clipped array       
        """
        grouped_data = self.data[start_idx:end_idx]

        return grouped_data

    def __getitem__(self, batch_idx):
        """Gets batch at position `batch_idx`.

        Arguments:
            batch_idx: position of the batch in the Sequence.

        Returns:
            A batch
        """
        x = np.zeros((self.batch_size,self.time_steps,self.num_cols))
        y1 = np.zeros((self.batch_size,1))
        y2 = np.zeros((self.batch_size,self.num_cols-1))

        idxi = batch_idx * self.batch_size
                
        for i in range(self.batch_size):
            x[i,:,:] = self.group_data(idxi,idxi + self.time_steps)
            out = self.data[idxi + self.time_steps]
            y1[i,:] = out[0]
            y2[i,:] = out[1:]
            idxi +=1
        
        
        return x , [y1,y2]


    def __len__(self):
        """Number of batch in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        
        return int(self.data.shape[0]/(self.batch_size+self.time_steps))+1

class LSTM_model(object):
    def __init__(self,**kwargs):
        
        self.sub_generator = PX4Generator()

        self.batch_size = 64
        self.time_steps = 200
        self.validation_split = 0.13           
        self.features = self.sub_generator.num_cols 
        self.epochs = 10
        self.workers = 2
        self.hidden = 500
        self.dropout = 0.5
        self.__dict__.update(kwargs)

        self.callbacks = list()
        self.props = {
            "batch_size": self.batch_size,
            "features": self.features,
            "time_steps":self.time_steps,
            "validation_split":self.validation_split
        }
        
    def add_callback(self,callback):
        """Adds a callback to the list of callbacks

        Argument:
            callback: callback to be added to the current callbacks list
        """
        
        self.callbacks.append(callback)
    
    def model(self):
        """Creates a neural network model
        
        Returns:
            A model
        """
        print("creating the model...")
        inputs = K.layers.Input(shape=(self.time_steps,self.features))
        x = K.layers.CuDNNLSTM(self.hidden,return_sequences=True)(inputs)
        x = K.layers.CuDNNLSTM(self.hidden)(x)
        x = K.layers.Dropout(self.dropout)(x)

        time = K.layers.Dense(1,activation="linear",name="time")(x)
        data = K.layers.Dense(self.features-1,activation="sigmoid",name = "data")(x)
        
        model = K.models.Model(
            inputs = inputs,
            outputs = [time,data],
            name = "PX4_NN"
        )
        #Prepare losses
        losses = {
            "data":"binary_crossentropy",
            "time":"mse"
        }
        
        # Prepare loss weights.
        lossWeights = {
            "data":1.0,
            "time":1.0
        }
        #Configures the model for training.
        model.compile(
            loss = losses,
            loss_weights = lossWeights,
            optimizer = "adadelta",
            metrics = ["accuracy"]
        )
        model.summary()
        return model

    def fit(self):
        """Trains the model for a fixed number of epochs

        Returns:
            A History object
        """
        model = self.model()
        train = PX4Generator(train=True, **self.props)
        valid = PX4Generator(train=False,**self.props)
        hist = model.fit_generator(
            generator=train,
            validation_data=valid,
            epochs=self.epochs,
            use_multiprocessing=True,
            workers=self.workers,
            callbacks=self.callbacks,
            verbose=1,
            shuffle=True
        )

        model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs/%s/%s/result.hdf5" % (str(os.path.basename(os.path.dirname(os.path.dirname(self.sub_generator.path))))[-5:],PID)))
        return hist


def main(argv):
    LSTM = LSTM_model(
        hidden=int(argv[1]), 
        time_steps=int(argv[2]), 
        dropout=float(argv[3]),
    )  
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs/%s/%s" % (str(os.path.basename(os.path.dirname(os.path.dirname(LSTM.sub_generator.path))))[-5:],PID))):
        try:
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs/%s/%s" % (str(os.path.basename(os.path.dirname(os.path.dirname(LSTM.sub_generator.path))))[-5:],PID)),0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tensorboard/%s/%s" % (str(os.path.basename(os.path.dirname(os.path.dirname(LSTM.sub_generator.path))))[-5:],PID))):
        try:
            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tensorboard/%s/%s" % (str(os.path.basename(os.path.dirname(os.path.dirname(LSTM.sub_generator.path))))[-5:],PID)),0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    LSTM.add_callback(K.callbacks.ModelCheckpoint(os.path.join(os.path.dirname(os.path.abspath(__file__)),"logs/%s/%s/{epoch:02d}-{val_data_acc:.2f}.hdf5" % (str(os.path.basename(os.path.dirname(os.path.dirname(LSTM.sub_generator.path))))[-5:],PID)), verbose=1, save_best_only=False, mode='max', monitor='val_loss'))
    LSTM.add_callback(K.callbacks.TensorBoard(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tensorboard/%s/%s" % (str(os.path.basename(os.path.dirname(os.path.dirname(LSTM.sub_generator.path))))[-5:],PID))))
    LSTM.add_callback(K.callbacks.EarlyStopping(verbose=1, monitor='val_loss'))
    LSTM.add_callback(K.callbacks.TerminateOnNaN())
    
    hist = LSTM.fit()
    print(hist)


if __name__ == "__main__":
    global PID

    import multiprocessing
    multiprocessing.set_start_method('spawn', True)


    for L in [100]: 
        for N in [50]: 
            K.backend.clear_session()
            inp = ['',L,N,0.1]
            PID = "%d-%d-drop0.1" % (L,N)
            main(inp)

            
            