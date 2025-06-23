import promptbench as pb

# print all supported datasets in promptbench
print('All supported datasets: ')
print(pb.SUPPORTED_DATASETS)

# load a dataset, sst2, for instance.
# if the dataset is not available locally, it will be downloaded automatically.
dataset_name = "gsm8k"
dataset = pb.DatasetLoader.load_dataset(dataset_name)

# print the first 3 examples
dataset[:3]
