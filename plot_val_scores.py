import sys
import matplotlib.pyplot as plt


nohup_file = sys.argv[1]
metric2scores = {}
epochs = []
with open(nohup_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if '. Best epoch is' in line:
            metric = line.split('Validation: ')[1].split(' ')[0]
            if metric not in metric2scores.keys():
                metric2scores[metric] = []
            score = float(line.split('. Best epoch is')[0].split(' ')[-1])
            epoch = int(line.split('for epoch-')[1].split(' ')[0])
            epochs.append(epoch)
            metric2scores[metric].append(score)
plt.figure()
plt.legend(metric2scores.keys())
plt.xlabel('Epochs')
plt.ylabel('Metric Scores')
for metric, scores in metric2scores.items():
    plt.plot(epochs, scores)
plt.savefig(nohup_file.replace('nohup.out.', 'plot_').replace('nohup.out', 'plot_'))
plt.show()

