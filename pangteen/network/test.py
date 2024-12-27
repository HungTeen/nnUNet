
if __name__ == '__main__':
    dict = {
        'one': 1,
        'two': 2,
        'three': {
            'four': 4,
            'five': {
                'six': 6,
                'seven': 7
            }
        },
    }
    three = dict['three']
    five = three['five']
    five.pop('six')
    three['five'] = five
    dict['three'] = three
    print(dict)