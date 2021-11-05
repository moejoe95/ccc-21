import os

import numpy as np

file1 = open(os.path.join('..', 'data', 'level_1', 'level_1_4.csv'), 'r')

lines = file1.readlines()

num_samples = int(lines[0].strip())
threshold = int(lines[1].strip())

samples = []

results = []

for n, line in enumerate(lines[2:]):
    line = line.strip()
    sample = line.split(',')
    sample = [int(x) for x in sample]
    samples.append(sample)

    sample_np = np.asarray(sample)
    sample_np = sample_np[sample_np != 0]

    if np.mean(sample_np) > threshold:
        results.append(True)
    else:
        results.append(False)

# print(len([x for x in results if x == 1]))
# for result in results:
#     print(result)

output_file = "output.csv"
result = np.array(results)

f = open(output_file, "w")
f.write(f"{result.sum()}\n")
for image in result:
    f.write(f"{int(image)}\n")

f.close()