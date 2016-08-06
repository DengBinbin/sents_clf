from collections import Counter
from path import *
import re
import codecs
import cPickle
regex_enter = re.compile('\n')
file_path = os.path.join(data_path,'51job2.txt')
result_path =  os.path.join(data_path,'char_count.txt')
with codecs.open(file_path,'rb','utf-8') as fi:
	content = re.sub(regex_enter,'',fi.read())
	char_count = Counter(content)

most_common = char_count.most_common()

input_pkl = file(data_path+'all_input_char.pkl', "a")
cPickle.dump(char_count, input_pkl)

with codecs.open(result_path,'wb','utf-8') as fo:	
	for item in most_common:
		fo.writelines( item[0]+'   '+str(item[1])+u'\n')