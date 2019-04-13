import sklearn
import glob

def ViewFiles():
    list_of_Negfiles = glob.glob('./corpus/neg/*.txt')
    list_of_Posfiles = glob.glob('./corpus/pos/*.txt')
    samples = []
    labels = []
    for i in list_of_Negfiles:
        f = open(i)
        samples.append(f.read())
        labels.append(0)
    for i in list_of_Posfiles:
        f = open(i)
        samples.append(f.read())
        labels.append(1)
    return samples, labels
    # print("Negatives: ", list_of_Negfiles)
    # print("Positives: ", list_of_Posfiles)


def main():
    print(ViewFiles()[999][1])

if __name__ == "__main__":
    main()