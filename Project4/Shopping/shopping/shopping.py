import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def monthParser(inputStr):
    inputStr = inputStr.lower()
    if inputStr == "jan":
        return 0
    elif inputStr == "feb":
        return 1
    elif inputStr == "mar":
        return 2
    elif inputStr == "apr":
        return 3
    elif inputStr == "may":
        return 4
    elif inputStr == "june":
        return 5
    elif inputStr == "jul":
        return 6
    elif inputStr == "aug":
        return 7
    elif inputStr == "sep":
        return 8
    elif inputStr == "oct":
        return 9
    elif inputStr == "nov":
        return 10
    elif inputStr == "dec":
        return 11

def visitorParser(inputStr):
    if inputStr == "Returning_Visitor":
        return 1
    else:
        return 0

def booleanParser(inputStr):
    if inputStr == "TRUE":
        return 1
    else:
        return 0

def oneHotEncode_Evi(row):
    adminstrative = int(row[0])
    adminstrative_duration = float(row[1])
    informational = int(row[2])
    informational_duration = float(row[3])
    productRelated = int(row[4])
    productRelated_duration = float(row[5])
    bounceRates = float(row[6])
    exitRates = float(row[7])
    pageValues = float(row[8])
    specialDay = float(row[9])
    month = monthParser(row[10])
    operatingSystem = int(row[11])
    brower = int(row[12])
    region = int(row[13])
    trafficType = int(row[14])
    visitorType = visitorParser(row[15])
    weedend = booleanParser(row[16])
    return [
        adminstrative, adminstrative_duration, informational, informational_duration, 
        productRelated, productRelated_duration, bounceRates, exitRates, pageValues, 
        specialDay, month, operatingSystem, brower, region, trafficType, visitorType, weedend
    ]

def oneHotEncode_labels(row):
    return booleanParser(row)

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open("./shopping.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        evidence_raw = []
        labels_raw = []
        for row in reader:
            evidence_raw.append(row[:-1])
            labels_raw.append(row[-1])
    evidence = []
    labels = []
    for row1, row2 in zip(evidence_raw, labels_raw):
        evidence.append(oneHotEncode_Evi(row1))
        labels.append(oneHotEncode_labels(row2))
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    TP = 0
    actualP = 0
    TN = 0
    actualN = 0
    for label, prediction in zip(labels, predictions):
        if label ==1:
            actualP +=1
            if prediction == 1:
                TP +=1
        else:
            actualN +=1
            if prediction ==0:
                TN +=1
                
    sensitivity = float(TP/actualP)
    specificity = float(TN/actualN)
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
