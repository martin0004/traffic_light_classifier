import copy
import cv2
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psub
import sys
from typing import Iterable, List, Union


#######################################################################
#                                                                     #
#                    GLOBAL VARIABLES                                 #
#                                                                     #
#######################################################################  


CLASSES = ["red", "yellow", "green"]


#######################################################################
#                                                                     #
#                    DATALOADER                                       #
#                                                                     #
#######################################################################  


class DataLoader(object):
    """ Class which initializes a dataset from a directory. """
    
    def __init__(self, classes: List[str]):         
        
        self.classes = classes
    
    def one_hot_encode_label(self, label: str) -> np.array:
        
        # Change label to list of Booleans
        #
        # "red"    -> [True, False, False]
        # "yellow" -> [False, True, False]
        # "green"   -> [False, False, True]
        
        label = [ label == c for c in self.classes]
                
        # Change vector of booleans to vector of integers
 
        # [True, False, False] -> [1, 0, 0]
        # [False, True, False] -> [0, 1, 0]
        # [False, False, True] -> [0, 0, 1]
        
        label = np.array(label, dtype=int)    
        
        return label    
    
    def load_image(self, img_path: str) -> pd.DataFrame:
        """ Create dataframe from unlabelled image. """
        
        # Dictionnary which will hold image.       
        dict_rgb = dict()
        dict_rgb["rgb"] = []            # image in RGB format
        
        # Fill dictionnary        
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dict_rgb["rgb"].append(rgb)
                    
        # Turn dictionnary to DataFrame       
        df_rgb = pd.DataFrame(dict_rgb)
        
        # Return dataframe
        return df_rgb
    
    
    def load_dataset(self, root_dir: str) -> pd.DataFrame:
        """ Create labeled dataset from content of directory. """
        
        # This method will load images in root_dir
        # by doing the following.
        #
        # 1 - Each subdirectory in the root directory
        #     will be considred a class label.
        # 2 - Files directly below root_dir will be ignored.
        # 3 - Directories inside subdirectories will be ignored.
        #
        # root_dir/
        #    class1/
        #        image01.jpg
        #        image02.jpg
        #        directory/        # Will be ignored.
        #        image03.jpg
        #        (...)
        #    class2/
        #        image01.jpg
        #        image02.jpg
        #        (...) 
        #    (...)
        #    somefile.txt          # Will be ignored.
        
        # List of classes in this dataset
        # (each root_dir subdirectory is assumed a class).
        
        # Dictionnary which will hold images.       
        dict_rgb = dict()
        dict_rgb["rgb"] = []            # image in RGB format
        
        # Dictionnary which will hold labels
        dict_labels = dict()
        dict_labels["label"] = []   
        
        
        # Fill dictionnaries
        
        # Loop tru each subdirectory
        # (each subdirectory is a class name)
        
        for sub_dir in os.listdir(root_dir):
            
            sub_dir_path = os.path.join(root_dir, sub_dir)
            
            # Check if this is a directory
            # (ignores files)
            
            if os.path.isdir(sub_dir_path):
                
                # Loop tru each file
                
                for f in os.listdir(sub_dir_path):
                    f_path = os.path.join(sub_dir_path, f) 
                    
                    # Check if this is a file
                    # (ignores directories)
                    
                    if os.path.isfile(f_path):
                            
                        bgr = cv2.imread(f_path)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        dict_rgb["rgb"].append(rgb)
                          
                        dict_labels["label"].append(sub_dir)
                    
        # Turn dictionnaries to DataFrames           
        
        df_rgb = pd.DataFrame(dict_rgb)
        df_labels = pd.DataFrame(dict_labels)
        
        # One-hot encode labels
        
        df_labels["label"] = df_labels["label"].apply(lambda x: self.one_hot_encode_label(x))
        
        return df_rgb, df_labels


#######################################################################
#                                                                     #
#                    CLASSIFIER COMPONENTS                            #
#                                                                     #
#######################################################################  
   

class DataPreprocessor(object):
    
    def __init__(self, img_size: int):
        
        # Side of square image after pre-processing [pixels]
        self.img_size = img_size

    def resize_image(self, rgb: np.array) -> np.array:
        
        # Standardize image sizes
        
        rgb_resized = cv2.resize(rgb, (self.img_size, self.img_size))
        
        return rgb_resized
    
    def run(self, df_in: pd.DataFrame) -> pd.DataFrame:
        
        # Copy dataset        
        df_out = copy.deepcopy(df_in)
        
        # Process dataset        
        df_out["rgb"]   = df_out["rgb"].apply(lambda x: self.resize_image(x))
 
        return df_out


class FeatureExtractor(object):
    
    def __init__(self, crop_left: int, crop_top: int):
        
        self.crop_left = crop_left   # Number of pixels for left & right crop
        self.crop_top  = crop_top    # Number of pixels for top & bottom crop

    def crop_image(self, rgb: np.array) -> np.array:
        
        rgb_cropped = rgb[self.crop_top:-self.crop_top,
                          self.crop_left:-self.crop_left,
                         :]
        
        return rgb_cropped

    def normalize_vector(self, v: np.array) -> np.array:
        
        norm = np.linalg.norm(v)
        
        if norm > 0:        
            v_normalized = v / norm            
        else:            
            v_normalized = np.copy(v)
  
        # Limit values to 3 decimals.
        # This allowed to print readable dataframes
        # and this precision was good enough for this project.
        
        v_normalized = np.round(v_normalized, decimals=3)
        
        return v_normalized    

    def get_brightness_vector(self, rgb: np.array) -> np.array:
        """ Sum brightness values in image. """
        
        brightness_vector = np.array([0,0,0])
        
        rgb_cropped = self.crop_image(rgb)
        hsv_cropped = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2HSV)
        v_cropped = hsv_cropped[:,:,2]
        
        m = v_cropped.shape[0]        
        a = int(m/3)        
        b = int(2*a)
        
        brightness_vector[0] = np.sum(v_cropped[:a,:])
        brightness_vector[1] = np.sum(v_cropped[a:b,:])
        brightness_vector[2] = np.sum(v_cropped[b:,:])
    
        return brightness_vector
    
    def apply_hsv_mask(self, rgb: np.array, h_min: int, h_max: int, v_min: int, v_max: int) -> np.array:
        
        hsv = convert_rgb_to_hsv(rgb)
        h = hsv[:,:,0]
        
        lower = np.array([h_min, 0, v_min])
        upper = np.array([h_max, 255, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        rgb_masked = np.copy(rgb)
        rgb_masked[mask == 0] = [0, 0, 0]
        
        return rgb_masked        
 
    def apply_green_mask(self, rgb: np.array) -> np.array:
        
        return self.apply_hsv_mask(rgb, 86, 94, 150, 255)
    
    def apply_yellow_mask(self, rgb: np.array) -> np.array:
        
        return self.apply_hsv_mask(rgb, 14, 32, 200, 255)

    def apply_red_mask(self, rgb: np.array) -> np.array:
        
        # Red pixels hue values are divided into 2 clusters:
        # - one cluster has low hue values,
        # - another has higher values
        #
        # Images are obtained by masking for both ranges of hue values.
        # Both images are then combined.
        
        rgb_masked_1 = self.apply_hsv_mask(rgb,   0,   7, 100, 255)        
        rgb_masked_2 = self.apply_hsv_mask(rgb, 134, 179, 100, 255)
        rgb_combined = np.maximum(rgb_masked_1, rgb_masked_2)
        
        return rgb_combined
    
    def get_color_vector(self, rgb: np.array) -> np.array:
        """ Count number of red-yellow-green pixels in image. """
        
        color_vector = np.array([0,0,0])
        
        rgb_cropped = self.crop_image(rgb)
        
        color_vector[0] = np.count_nonzero(self.apply_red_mask(rgb_cropped).sum(axis=2))
        color_vector[1] = np.count_nonzero(self.apply_yellow_mask(rgb_cropped).sum(axis=2))
        color_vector[2] = np.count_nonzero(self.apply_green_mask(rgb_cropped).sum(axis=2))
        
        return color_vector    
        
    def run(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """ Extracts features from dataset. """
        
        # Initialize output dataframe
        df_out = pd.DataFrame()
        
        # Brightness vector        
        df_out["X1"] = df_in["rgb"].apply(lambda x: self.get_brightness_vector(x))
        df_out["x1"] = df_out["X1"].apply(lambda x: self.normalize_vector(x))
        
        # Color vector
        df_out["X2"] = df_in["rgb"].apply(lambda x: self.get_color_vector(x))
        df_out["x2"] = df_out["X2"].apply(lambda x: self.normalize_vector(x))
        
        # Return dataframe
        return df_out


class MachineLearningModel(object):
    
    def __init__(self, w1: float, w2: float):
        
        self.w1 = w1   # Weight for feature 1
        self.w2 = w2   # Weight for feature 2

    def get_max_score(self, v: np.array) -> np.array:
        """ Set highest value of vector to 1, others to 0. """
        
        max_score = np.zeros_like(v, dtype = int)
        
        # Do something of at least 1 item in v
        # different from 0.
        
        if np.any(v):
            
            v_max = np.amax(v)                  # Max value in v
            index_v_max = np.where(v == v_max)  # Index of item with max value in v
            
            max_score[index_v_max] = 1
        
        return max_score
        
       
    def run(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """ Predict input labels using linear equation
        
            prediction = max_score(w1 * x1 + w2 * x2)
        
        """
        
        # Initialize output dataframe
        df_out = pd.DataFrame()
        
        # Prediction        
        df_out["y_bar"] = self.w1*df_in["x1"] + self.w2*df_in["x2"]
        df_out["y_bar"] = df_out["y_bar"].apply(lambda x: self.get_max_score(x))
        
        # Return dataframe
        return df_out


class Classifier(object):
    
    def __init__(self, img_size: int,
                       crop_left: int,
                       crop_top: int,
                       w1: float,
                       w2: float):
        
        self.data_preprocessor = DataPreprocessor(img_size)
        self.feature_extractor = FeatureExtractor(crop_left, crop_top)
        self.machine_learning_model = MachineLearningModel(w1, w2)
        
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Run classification pipeline
        
        df = self.data_preprocessor.run(df)
        df = self.feature_extractor.run(df)
        df = self.machine_learning_model.run(df)
        
        # Output predictions
        
        return df

    
#######################################################################
#                                                                     #
#                    EVALUATOR                                        #
#                                                                     #
#######################################################################     


class Evaluator(object):

    
    def __init__(self):
        
        # No parameters to initialize for this class...        
        pass

    
    def get_labels_predicted_right(self,
                                   true_labels: Iterable[Iterable],
                                   predicted_labels: Iterable[Iterable]) -> pd.DataFrame:
        """Identify right predictions."""
        
        df_in = pd.DataFrame({"true_labels": true_labels,
                              "predicted_labels": predicted_labels})
        
        indices_out = [index for index, s in df_in.iterrows() if (s["true_labels"] == s["predicted_labels"]).all()]        
        df_out = df_in.iloc[indices_out, :]        
        
        return df_out    
    
    
    def get_labels_predicted_wrong(self,
                                   true_labels: Iterable[Iterable],
                                   predicted_labels: Iterable[Iterable]) -> pd.DataFrame:
        """Identify wrong predictions."""
        
        df_in = pd.DataFrame({"true_labels": true_labels,
                              "predicted_labels": predicted_labels})
        
        indices_out = [index for index, s in df_in.iterrows() if not (s["true_labels"] == s["predicted_labels"]).all()]        
        df_out = df_in.iloc[indices_out, :]        
        
        return df_out


    def get_reds_predicted_green(self,
                                 true_labels: Iterable[Iterable],
                                 predicted_labels: Iterable[Iterable]) -> pd.DataFrame:
        """Identify red lights predicted green."""
        
        red = np.array([1,0,0])
        green = np.array([0,0,1])    
        
        df_in = pd.DataFrame({"true_labels": true_labels,
                              "predicted_labels": predicted_labels})
        
        indices_out = [index for index, s in df_in.iterrows()
                       if (s["true_labels"] == red).all() and (s["predicted_labels"] == green).all()]
        
        df_out = df_in.iloc[indices_out, :]
        
        return df_out


    def get_accuracy(self,
                     true_labels: Iterable[Iterable],
                     predicted_labels: Iterable[Iterable]) -> float:
        """ Derive predictions accuracy. """
        
        accuracy = 0          # Accuracy
        
        n_total = len(true_labels)
        n_right = len(self.get_labels_predicted_right(true_labels, predicted_labels))
        
        if n_total != 0:
            accuracy = n_right / n_total
            
        return accuracy

    
    def run(self,
            true_labels: Iterable[Iterable],
            predicted_labels: Iterable[Iterable]) -> dict:
        """Derive evaluation metrics specific to this project."""
        
        # Initialize metrics        
        metrics = dict()
        
        # Derive metrics
        metrics["accuracy"] = self.get_accuracy(true_labels, predicted_labels)
        metrics["labels_predicted_right"] = self.get_labels_predicted_right(true_labels, predicted_labels)
        metrics["labels_predicted_wrong"] = self.get_labels_predicted_wrong(true_labels, predicted_labels)
        metrics["reds_predicted_green"] = self.get_reds_predicted_green(true_labels, predicted_labels)
        
        return metrics


#######################################################################
#                                                                     #
#                    VISUALIZATION TOOLS                              #
#                                                                     #
#######################################################################    
       
    
def print_images(dt: pd.DataFrame, *indices: int) -> None:
    """Display images from a dataset column 'rgb'."""
    
    # Figure parameters
    
    subplot_width = 400
    subplot_height = subplot_width
    
    n_cols = 3
    n_rows = int((len(indices)-1)/n_cols) + 1
    
    figure_width = n_cols * subplot_width
    figure_height = n_rows * subplot_height
    
    # Create subplots
    
    fig = psub.make_subplots(rows=n_rows, cols=n_cols)
    
    for i in range(len(indices)):
        
        image_index = indices[i]
        
        # Equations for calculating row and col below from here:
        # https://stackoverflow.com/questions/64268081/creating-a-subplot-of-images-with-plotly
        row = int( i/n_cols ) + 1
        col = i % n_cols + 1
        
        rgb = dt.loc[image_index]["rgb"]
    
        trace = go.Image(z=rgb)
        fig.add_trace(trace, row=row, col=col)
    
    
    # Show figure
    
    fig.layout.height = figure_height
    fig.layout.width = figure_width  
    
    fig.show()    

    
def print_pre_processed_image(rgb_old: np.array, img_size: int) -> None :
    """Print image before and after preprocessing."""
    
    # Process data                                      
                                              
    p = DataPreprocessor(img_size)
    
    rgb_new   = p.resize_image(rgb_old)                                                                                 
                                              
    # Figure parameters
    
    figure_title = "Before / After Pre-processing"
    
    subplot_width = 400
    subplot_height = subplot_width
    
    n_cols = 2
    n_rows = 1
    
    figure_width = n_cols * subplot_width
    figure_height = n_rows * subplot_height
    
    # Subplot titles
        
    subplot_title_old = "Before"
    subplot_title_new = "After"
        
    subplot_titles = [subplot_title_old, subplot_title_new]
    
    # Print figure
    
    fig = psub.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
     
    trace = go.Image(z=rgb_old)
    fig.add_trace(trace, row=1, col=1)
    
    trace = go.Image(z=rgb_new)
    fig.add_trace(trace, row=1, col=2)
    
    fig.layout.height = figure_height
    fig.layout.width = figure_width  
    
    fig.layout.title = figure_title
    fig.layout.title.x = 0.5
    
    fig.show()

    
def print_brightness_vector(rgb: np.array,
                            crop_left: int,
                            crop_top: int) -> None:
    """Print brightness vector for a single image."""
    
    # Process data
    
    f = FeatureExtractor(crop_left, crop_top)
    
    rgb_cropped = f.crop_image(rgb)
    X = f.get_brightness_vector(rgb)  # Unormalized feature
    x = f.normalize_vector(X)         # Normalized feature
    
    # Figure parameters
    
    figure_title = "Brightness Vector"
    
    n_cols = 5
    n_rows = 1
    
    # Subplot titles
    
    subplot_titles = ["Original Image",
                      "Cropped Image",
                      "Sum of Brightness Values <br>",
                      "Sum of Brightness Values <br>",
                      "Sum of Brightness Values <br> (Normalized)"]
       
    # Print figure

    fig = psub.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    
    trace = go.Image(z=rgb)
    fig.add_trace(trace, row=1, col=1)
    
    trace = go.Image(z=rgb_cropped)
    fig.add_trace(trace, row=1, col=2)
    
    y = list(range(X.shape[0]))
    trace = go.Bar(y=y, x=X, orientation = "h")
    fig.layout.yaxis3.autorange = "reversed"
    fig.add_trace(trace, row=1, col=3)
    
    img = np.full((3,1,3), 255, dtype=np.uint8) # dummy 3-pixel image, for representing vector
    
    trace = go.Image(z=img)
    fig.add_trace(trace, row=1, col=4)
    fig.layout.xaxis4.showticklabels = False
    fig.layout.yaxis4.showticklabels = False
    fig.add_annotation(xref="x4", yref="y4", x=0, y=0, text=str(X[0]), showarrow=False, font_size = 15, font_color="black")
    fig.add_annotation(xref="x4", yref="y4", x=0, y=1, text=str(X[1]), showarrow=False, font_size = 15, font_color="black")
    fig.add_annotation(xref="x4", yref="y4", x=0, y=2, text=str(X[2]), showarrow=False, font_size = 15, font_color="black")
    
    trace = go.Image(z=img)
    fig.add_trace(trace, row=1, col=5)
    fig.layout.xaxis5.showticklabels = False
    fig.layout.yaxis5.showticklabels = False
    fig.add_annotation(xref="x5", yref="y5", x=0, y=0, text=str(x[0]), showarrow=False, font_size = 15, font_color="black")
    fig.add_annotation(xref="x5", yref="y5", x=0, y=1, text=str(x[1]), showarrow=False, font_size = 15, font_color="black")
    fig.add_annotation(xref="x5", yref="y5", x=0, y=2, text=str(x[2]), showarrow=False, font_size = 15, font_color="black")
    
    fig.layout.title = figure_title
    fig.show()  

    
def print_color_vector(rgb: np.array,
                       crop_left: int,
                       crop_top: int) -> None:
    """Color vector for a single image."""
    
    # Process data
    
    f = FeatureExtractor(crop_left, crop_top)
    
    rgb_cropped = f.crop_image(rgb)
    
    rgb_masked_green = f.apply_green_mask(rgb_cropped)
    rgb_masked_yellow = f.apply_yellow_mask(rgb_cropped)
    rgb_masked_red = f.apply_red_mask(rgb_cropped)
    
    X = f.get_color_vector(rgb)  # Unormalized feature
    x = f.normalize_vector(X)    # Normalized feature
    
    # Figure parameters
    
    figure_title = "Color Vector"
        
    n_cols = 5
    n_rows = 3
    
    # Subplot titles
    
    subplot_titles = ["",
                      "",
                      "Masked Image <br> (Red)",
                      "",
                      "",
                      "Original Image",
                      "Cropped Image",
                      "Masked Image <br> (Yellow)",
                      "Sum of Red/Yellow/Green <br> Pixels",
                      "Sum of Red/Yellow/Green <br> Pixels (Normalized)",
                      "",
                      "",
                      "Masked Image <br> (Green)",
                      "",
                      ""]
       
    # Print image

    fig = psub.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    
    trace = go.Image(z=rgb)
    fig.add_trace(trace, row=2, col=1)
    
    trace = go.Image(z=rgb_cropped)
    fig.add_trace(trace, row=2, col=2)
    
    trace = go.Image(z=rgb_masked_red)
    fig.add_trace(trace, row=1, col=3)
    
    trace = go.Image(z=rgb_masked_yellow)
    fig.add_trace(trace, row=2, col=3)
    
    trace = go.Image(z=rgb_masked_green)
    fig.add_trace(trace, row=3, col=3)
    
    img = np.full((3,1,3), 255, dtype=np.uint8) # dummy 3-pixel image, for representing vector
    
    trace = go.Image(z=img)
    fig.add_trace(trace, row=2, col=4)
    fig.layout.xaxis9.showticklabels = False
    fig.layout.yaxis9.showticklabels = False
    fig.add_annotation(xref="x9", yref="y9", x=0, y=0, text=str(X[0]), showarrow=False, font_size = 15, font_color="black")
    fig.add_annotation(xref="x9", yref="y9", x=0, y=1, text=str(X[1]), showarrow=False, font_size = 15, font_color="black")
    fig.add_annotation(xref="x9", yref="y9", x=0, y=2, text=str(X[2]), showarrow=False, font_size = 15, font_color="black")

    trace = go.Image(z=img)
    fig.add_trace(trace, row=2, col=5)
    fig.layout.xaxis10.showticklabels = False
    fig.layout.yaxis10.showticklabels = False
    fig.add_annotation(xref="x10", yref="y10", x=0, y=0, text=str(x[0]), showarrow=False, font_size = 15, font_color="black")
    fig.add_annotation(xref="x10", yref="y10", x=0, y=1, text=str(x[1]), showarrow=False, font_size = 15, font_color="black")
    fig.add_annotation(xref="x10", yref="y10", x=0, y=2, text=str(x[2]), showarrow=False, font_size = 15, font_color="black")    
    
    fig.layout.title = figure_title
    fig.layout.height = 1000
    fig.show()  
    

def convert_rgb_to_hsv(rgb):
    """Convert an rgb image to hsv colorspace."""
    
    # Note. rgb and hsv images must be M x N x 3.
    
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Additional line here to convert to int32
    # This avoids uint8 to be converted to int8
    # by some functions, thus avoiding hsv values
    # greater than 128 becoming negative...
    hsv = np.array(hsv, np.int32)
    
    return hsv


def plot_hsv_ranges(df_rgb: pd.DataFrame, csv: str) -> None:

    # Load pixel x-y coordinates
    df_px = pd.read_csv(csv)
    
    # Create Dictionnary which will hold hsv values
    
    dict_hsv = dict()
    
    dict_hsv["label"] = []
    dict_hsv["h"] = []
    dict_hsv["s"] = []
    dict_hsv["v"] = []
    
    # Fill dict_hsv
    
    for i, s in df_px.iterrows():
        
        label = s["label"]
        x = s["x"]
        y = s["y"]
        
        # Retrieve rgb image
        rgb = df_rgb.loc[s["img_index"], "rgb"]
        
        # Convert image to hsv
        hsv = convert_rgb_to_hsv(rgb)
        
        # Retrieve h value of pixels
        h = eval("hsv[" + x + "," + y + ", 0]").flatten()
        s = eval("hsv[" + x + "," + y + ", 1]").flatten()
        v = eval("hsv[" + x + "," + y + ", 2]").flatten()
        
        # Add hsv values in dict_hsv        
        for i in range(len(h)):
        
            dict_hsv["label"].append(label)
            dict_hsv["h"].append(h[i])
            dict_hsv["s"].append(s[i])
            dict_hsv["v"].append(v[i])
     
    # Create Dataframe from dict        
    df_hsv = pd.DataFrame(dict_hsv)
    
    # Plot violin charts of hsv value ranges
    
    for channel in ["h", "s", "v"]:
    
        fig = go.Figure()    
    
        # Create a separate trace for "green", "yellow" and "red" label values,
        # each with an appropriate color
    
        for label in df_hsv["label"].unique():

            # Dataframe containing only data for "green", "yellow" or "red"
            df_label = df_hsv[df_hsv["label"]==label]
        
            trace = go.Violin(x=df_label[channel])
            trace.name = label
            trace.points="all"
            trace.fillcolor=label
            trace.line.color=label
            trace.orientation = "h"
        
            fig.add_trace(trace)
    
        fig.layout.title = channel + " Value Ranges"
        fig.layout.height = 500
        fig.layout.width = 1000
        fig.layout.xaxis.range = [-10,300]
        fig.show()

    
def print_metrics(metrics) -> None:
    
    # Process data
    
    acc = metrics["accuracy"]
    n_right = len(metrics["labels_predicted_right"])
    n_wrong = len(metrics["labels_predicted_wrong"])
    n_reds_predicted_green = len(metrics["reds_predicted_green"])
    
    n_total = n_right + n_wrong
    
    # Figure parameters
    
    figure_title = "Evaluation Metrics"
    
    subplot_width = 400
    subplot_height = subplot_width
    
    n_cols = 2
    n_rows = 1
    
    figure_width = n_cols * subplot_width
    figure_height = n_rows * subplot_height    
       
    # Print figure
    
    fig = psub.make_subplots(rows=n_rows, cols=n_cols,
                             specs=[ [{"type": "table"}, {"type": "bar"}]])
    
    trace = go.Table()
    trace.header.values = ["Metric","Value"]
    trace.header.font.size = 16
    col1 = [ "Accuracy", "Reds Predicted Green"]
    col2 = [ format(acc, ".3f"), n_reds_predicted_green ]
    trace.cells.values  = [ col1, col2 ]
    trace.cells.font.size = 16
    trace.cells.height = 24
    trace.columnwidth = [200, 100]
    fig.add_trace(trace, row=1, col=1)    
      
    trace = go.Bar()
    trace.x = ["Predicted Right", "Predicted Wrong"]
    trace.y = [ n_right, n_wrong ]
    trace.marker.color = "blue"
    fig.add_trace(trace, row=1, col=2)
    
    fig.layout.title = figure_title
    fig.layout.width = figure_width
    fig.layout.height = figure_height
    fig.show()
    
    # Print misclassified images
    
    print("WRONG PREDICTIONS")
    print()
    print(metrics["labels_predicted_wrong"])
    print()
    print("RED LIGHTS CLASSIFIED AS GREEN")
    print()
    print(metrics["reds_predicted_green"])

    
#######################################################################
#                                                                     #
#                    MAIN                                             #
#                                                                     #
#######################################################################   



def main(argv):
    
    # Get image name
    
    img = argv[0]
    
    # Classifier parameters obtained during training
    
    img_size = 32
    crop_left = 12
    crop_top = 2
    w1 = 2
    w2 = 1
    
    # Load image
    
    df_image = DataLoader(CLASSES).load_image(img)
    
    # Run classifier on image

    df_prediction = Classifier(img_size, crop_left, crop_top, w1, w2).run(df_image)
    
    # Replace one-hot encoded prediction by word
    
    encoded_label = df_prediction["y_bar"][0]    
    index_max = np.argmax(encoded_label)    
    label = CLASSES[index_max]
    
    # Print label. Voilà!
    
    print(label)
    
    
if __name__ == "__main__":
    
    main(sys.argv[1:])