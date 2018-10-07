import numpy
data = """151 6
495 15
924 16
1193 42
183 52
30 53
484 62
102 64
981 65
208 67
370 88
2313 90
239 92
175 95
1 99""".split("\n")
data = numpy.array([[float(v) for v in row.split()] for row in data])
#print(data.shape, data)
Ntotal = data[:,0].sum()
mockline = ''
for N, cls in data:
	mockline += '%.6f,' % (N * 1. / Ntotal)
mockline = "," + mockline[:-1] + '\n'
print(mockline)
fout = open('prediction_out_mock.csv', 'w')
f = open('test_set_metadata.csv', 'r')
f.readline()
fout.write('object_id,class_6,class_15,class_16,class_42,class_52,class_53,class_62,class_64,class_65,class_67,class_88,class_90,class_92,class_95,class_99\n')
for line in f:
	object_id = line.split(',')[0]
	fout.write(object_id)
	fout.write(mockline)
fout.close()
f.close()

