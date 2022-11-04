# -*- coding: utf-8 -*-
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

# from IPython.display import SVG
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
# import numpy
import pandas as pd
import matplotlib



def build_simplenet(input_shape=(64,64,3),n_output_classes=2):
    """
    """
    model = Sequential()
    
    # Convolution + Pooling #1
    model.add(Conv2D( 32, (3, 3), input_shape=input_shape,
                          activation = 'relu' ))        
    model.add( MaxPooling2D(pool_size = (2,2)))
    
    # Convolution + Pooling #2
    model.add(Conv2D( 32, (3, 3), activation = 'relu' ))        
    model.add( MaxPooling2D(pool_size = (2,2)))
    
    # Flattening
    model.add( Flatten() )
    
    # FC #1
    model.add( Dense( units = 128, activation = 'relu' ) )
    
    # Output Layer
    model.add( Dense( units = n_output_classes, activation = 'softmax' ) )   
    
    # Compile
    model.compile( 
        optimizer = 'adam', loss = 'categorical_crossentropy',
        metrics = ['accuracy'] )
    return model

def train_simplenet( model,
               target_size,
               dataset_path,
               training_path_prefix,
               test_path_prefix,                        
               history_file_path,
               history_filename,
               checkpoint_path,
               checkpoint_prefix,
               number_of_epochs,
               tensorboard_log_path,
               batch_size,
#                     class_weight

            ):
    """
        see: https://keras.io/preprocessing/image/
    """
    train_datagen = ImageDataGenerator( rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True )
        
    test_datagen = ImageDataGenerator(rescale=1./255)
    training_set_generator = train_datagen.flow_from_directory(
            dataset_path+training_path_prefix,
            target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
    test_set_generator = test_datagen.flow_from_directory(
            dataset_path+test_path_prefix,
            target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
    step_size_train=training_set_generator.n//training_set_generator.batch_size
    step_size_validation=test_set_generator.n//test_set_generator.batch_size

    check_pointer = ModelCheckpoint(
            checkpoint_path + '%s_weights.{epoch:02d}-{val_loss:.2f}.hdf5' % checkpoint_prefix, 
            monitor='val_loss', 
            mode='auto', 
            save_best_only=True
    )
    
    tensorboard_logger = TensorBoard( 
        log_dir=tensorboard_log_path, histogram_freq=0,  
          write_graph=True, write_images=True
    )
    tensorboard_logger.set_model(model)

    csv_logger = CSVLogger(filename=history_file_path+history_filename)
    history = model.fit_generator(
            training_set_generator,
            steps_per_epoch=step_size_train,
            epochs=number_of_epochs,
            validation_data=test_set_generator,
            validation_steps=step_size_validation,
            callbacks=[check_pointer, csv_logger,tensorboard_logger],
#         class_weight =class_weight

    )
    
def plot_learning_curves_from_history_file(filename):
    matplotlib.rcParams['figure.figsize'] = (10, 6)
    history = pd.read_csv(filename)
    hv = history.values
    epoch=hv[:,0]
    acc=hv[:,1]
    loss=hv[:,2]
    val_acc=hv[:,3]
    val_loss=hv[:,4]
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(epoch,acc,epoch,val_acc)
    axes[0].set_title('model accuracy')
    axes[0].grid(which="Both")
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'test'], loc='lower right')
    axes[1].plot(epoch,loss,epoch,val_loss)
    axes[1].set_title('model loss')
    axes[1].grid(which="Both")
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'test'], loc='upper center')
    return fig

if __name__ == "__main__":
    
    model = build_simplenet()
    model.summary()
    plot_model(model, show_shapes=True)
    MODEL_NAME="simplenet_cracks8020_1"
    train_simplenet(model,
            target_size=(64,64),
            dataset_path="cracks_splitted8020/",
            training_path_prefix="train_set",
            test_path_prefix="test_set",
            history_file_path="training_logs/",
            history_filename=MODEL_NAME+".csv",
            checkpoint_path="model-checkpoints/",
            checkpoint_prefix=MODEL_NAME,
            number_of_epochs=1, 
            tensorboard_log_path="tensorboard_logs/",
            batch_size = 1024
)


