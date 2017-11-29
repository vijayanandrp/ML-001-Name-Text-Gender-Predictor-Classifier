import random
from string import punctuation


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
        
        
def data_process(file_name='', gender='F', output_file='global_female_names_parsed.txt'):
    _content = ''
    _dict = AutoVivification()
    with open(file_name) as f:
        _content = f.readlines()
    
    for _value in _content:
        invalidChars = set(punctuation)
        _value = _value.strip()
        if any(char in invalidChars for char in _value):
            tmp = (_value.replace(invalidChar, ' ') for invalidChar in invalidChars if invalidChar in _value)
            _value = ''.join(tmp).strip()
            
        names = _value.split(' ')
        for name in names:
            name = name.strip()
            if name in _dict.keys():
                _dict[name]['count'] += random.randint(50, 100)
            else:
                _dict[name]['count'] = random.randint(100, 500)
                _dict[name]['gender'] = gender
                
    # import pprint
    # pprint.pprint(_dict)
    
    output_file = open(output_file, 'w')
    for _name in _dict.keys():
        output_file.write(_name + ',' + _dict[_name]['gender'] + ',' + str(_dict[_name]['count']) + '\n')
    output_file.close()
    del _dict

male_file = 'global_male.txt'
female_file = 'global_female.txt'
data_process(file_name=male_file, gender='M', output_file='global_male_names_parsed.txt')
data_process(file_name=female_file, gender='F', output_file='global_female_names_parsed.txt')
