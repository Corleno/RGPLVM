import pods
import numpy as np
import pandas as pd

if __name__ == "__main__":
	name = "Anuran_Genus"
	# Load Anuran Calls (MFCCs)
	data = pd.read_csv("../data/Frogs_MFCCs.csv")
	print(data.columns)
	Y = data.iloc[:,:22].values
	# labels = data["Family"].values
	labels = data["Genus"].values
	# labels = data["Species"].values
	print(Y.shape, labels.shape)