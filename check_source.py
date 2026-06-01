import inspect
import my_dataset

print("my_dataset loaded from:")
print(my_dataset.__file__)

# Show the EXACT source of SegDataset currently used
print("\nSegDataset code location:\n")
print(inspect.getsource(my_dataset.SegDataset))
