import csv

# inputcsv = '/media/Data/yangx/lung_seg/DataSet/luna16/LUNA16_Seg/noduleA.csv'
def detcsv(id):
    inputcsv = r'C:\Users\Robert\Desktop/noduleE.csv'
    with open(inputcsv, 'r') as f:
        print(inputcsv)
        r_csv = csv.reader(f)
        print(1)
        Row = []
        for row in r_csv:
            if id == row[1]:
                print(row)
                Row.append(row)
    return Row

# Row = detcsv('CT00236000A')
# print(Row)


