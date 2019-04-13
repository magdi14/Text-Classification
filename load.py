import sklearn
import glob

def ViewFiles():
    list_of_Negfiles = glob.glob('./corpus/neg/*.txt')
    list_of_Posfiles = glob.glob('./corpus/pos/*.txt')
    print("Negatives: ", list_of_Negfiles)
    print("Positives: ", list_of_Posfiles)


def main():
    ViewFiles()

if __name__ == "__main__":
    main()