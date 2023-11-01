import os
HOME = os.getcwd()
TEST = os.path.join(HOME, 'test_images')
for file in os.listdir(TEST):
    filename = file.split('.')[0] + '.txt'
    with open(os.path.join(TEST, filename), 'w') as txt_file:
        pass
