from digitstruct import DigitStructFile

file = DigitStructFile('data/train/digitStruct.mat')

print(file.getName(1003))
print(file.getDigitStructure(1003))