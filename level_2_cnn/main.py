import os

import numpy as np

train_file = open(os.path.join('..', 'data', 'level_2', 'train.csv'), 'r')

lines = train_file.readlines()

num_samples = int(lines[0].strip())

train_samples = []
labels = []
results = []

for n in range(num_samples):
    sample_line = lines[1 + n].strip()
    label_line = lines[1 + num_samples + n].strip()
    sample = sample_line.split(',')
    sample = [int(x) for x in sample]
    label = int(label_line)
    train_samples.append(sample)
    labels.append(label)

# TODO shuffle


print()


    # sample_np = np.asarray(sample)
    # sample_np = sample_np[sample_np != 0]
    #
    # if np.mean(sample_np) > threshold:
    #     results.append(True)
    # else:
    #     results.append(False)

# output_file = "output.csv"
# result = np.array(results)
#
# f = open(output_file, "w")
# f.write(f"{result.sum()}\n")
# for image in result:
#     f.write(f"{int(image)}\n")
#
# f.close()