import csv
def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def write_to_file(records):
    """
    Function to write a list of records to a file
    """
    with open('modified_annotation_train.txt', 'a+') as outfile:
        outfile.writelines(records)


f = open('annotation_train.txt')
all_records = f.readlines()
results = []
for index, line in enumerate(all_records):
    file_name = line.split(' ')[0][2:]
    groundtruth = file_name.split('_')[1]
    results.append(f"/home/ubuntu/shubham/aon/data/mnt/ramdisk/max/90kDICT32px/{file_name} {groundtruth}\n")
write_to_file(results)
