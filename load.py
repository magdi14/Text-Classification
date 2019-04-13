import sklearn
import glob

def ViewFiles():
    list_of_Negfiles = glob.glob('./corpus/neg/*.txt')
    list_of_Posfiles = glob.glob('./corpus/pos/*.txt')
    texts = []
    for i in list_of_Negfiles:
        f = open(i)
        texts.append((f.read(), 0))
    for i in list_of_Posfiles:
        f = open(i)
        texts.append((f.read(), 1))
    return texts
    # print("Negatives: ", list_of_Negfiles)
    # print("Positives: ", list_of_Posfiles)
    # return list_of_Posfiles,list_of_Negfiles


def main():
    print(ViewFiles()[999][1])

if __name__ == "__main__":
    main()