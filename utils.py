import matplotlib
from matplotlib import pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import classification_report, confusion_matrix

def results(y_test, pred_y):
    return classification_report(y_test, pred_y,output_dict=True,target_names=['Desmpleado','Empleado'])

